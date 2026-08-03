# -*- coding: utf-8 -*-
"""Persona profile extractor and store (scene-driven)."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

try:
    import json_repair  # type: ignore
except Exception:  # pragma: no cover - fallback if json_repair not available
    class _JsonRepairFallback:
        @staticmethod
        def loads(s):
            return json.loads(s)
    json_repair = _JsonRepairFallback()  # type: ignore

from T_mem.persona.common import (
    PROFILE_DEBUG,
    PERSONA_EXTRACT_MODEL,
    PERSONA_EXTRACT_TIMEOUT,
    PROFILE_AGGREGATION_MIN_COUNT,
    PROFILE_TA_MAX_ITEMS_PER_KEY,
    PROFILE_TA_ALLOWED_KEYS,
    PROFILE_IDENTITY_ALLOWED_KEYS,
    PROFILE_SHARED_ACTIVITIES_MAX,
)
from T_mem.prompts.persona_prompts import PROFILE_EXTRACT_PROMPT
from T_mem.llm.venus_provider import venus_chat

logger = logging.getLogger(__name__)


@dataclass
class PersonProfile:
    """One speaker's profile: 5 structured buckets plus a timeline."""
    person_name: str = ""
    identity: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    traits_and_attitudes: dict = field(default_factory=dict)
    relations: list = field(default_factory=list)
    aggregations: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "person_name": self.person_name,
            "identity": {k: (v if isinstance(v, str) else str(v))
                         for k, v in self.identity.items()},
            "preferences": {k: list(v) for k, v in self.preferences.items()},
            "traits_and_attitudes": {k: list(v) for k, v in self.traits_and_attitudes.items()},
            "relations": [dict(r) for r in self.relations],
            "aggregations": dict(self.aggregations),
            "timeline": [dict(e) for e in self.timeline],
        }

    @classmethod
    def from_json(cls, data: dict) -> "PersonProfile":
        if not isinstance(data, dict):
            raise ValueError(f"PersonProfile.from_json expects dict, got {type(data)}")

        def _downgrade_identity(raw):
            """identity: str -> str; list[str] -> last element."""
            out = {}
            for k, v in (raw or {}).items():
                if v is None or v == "":
                    continue
                if isinstance(v, list):
                    vals = [x for x in v if x is not None and x != ""]
                    if not vals:
                        continue
                    out[k] = vals[-1] if isinstance(vals[-1], str) else str(vals[-1])
                else:
                    out[k] = v if isinstance(v, str) else str(v)
            return out

        ta_raw = data.get("traits_and_attitudes")
        if isinstance(ta_raw, dict) and ta_raw:
            ta = {k: [x for x in (v or []) if x] if isinstance(v, list) else [v]
                  for k, v in ta_raw.items()}
        else:
            ta = {k: [] for k in PROFILE_TA_ALLOWED_KEYS}

        return cls(
            person_name=data.get("person_name", "") or "",
            identity=_downgrade_identity(data.get("identity")),
            preferences={k: list(v) for k, v in (data.get("preferences") or {}).items()},
            traits_and_attitudes=ta,
            relations=[dict(r) for r in (data.get("relations") or [])],
            aggregations=dict(data.get("aggregations") or {}),
            timeline=[dict(e) for e in (data.get("timeline") or [])],
        )

    def is_empty(self) -> bool:
        ta_has = any(v for v in (self.traits_and_attitudes or {}).values())
        return not (self.identity or self.preferences or ta_has
                    or self.relations or self.aggregations or self.timeline)


_FILENAME_INVALID_RE = re.compile(r"[\\/:*?\"<>|\s]+")


def _sanitize_filename(name: str) -> str:
    if not name:
        return "unknown"
    safe = _FILENAME_INVALID_RE.sub("_", name.strip())
    return safe or "unknown"


def _norm_str(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower()


class ProfileMemory:
    """Two-speaker profile store. Merges LLM deltas and persists to disk.

    The scheduler (``scheduler.py``) owns the driving loop and invokes
    ``extract_and_merge_for_window(...)`` with a rendered ``chat_current`` text;
    this class is a pure merger + persister with no cross-module coupling.
    """

    def __init__(
        self,
        speaker_a: str,
        speaker_b: str,
        store_root: str,
        conv_label: str,
    ):
        """
        Args:
            speaker_a / speaker_b: the two fixed participants.
            store_root: filesystem root for persistence. Profiles are saved
                under ``<store_root>/<conv_label>/<speaker>.json``; we keep a
                single terminal snapshot per conv (no timestamp prefix).
            conv_label: directory name under ``store_root`` (e.g. ``qa_conv0``).
        """
        if not speaker_a or not speaker_b:
            raise ValueError(
                f"ProfileMemory requires both speakers (got a={speaker_a!r}, b={speaker_b!r})"
            )
        if speaker_a == speaker_b:
            raise ValueError(f"speaker_a and speaker_b must differ (got {speaker_a!r})")

        self.speaker_a = speaker_a
        self.speaker_b = speaker_b
        self.store_root = store_root
        self.conv_label = conv_label

        self.profiles: Dict[str, PersonProfile] = {
            speaker_a: PersonProfile(person_name=speaker_a),
            speaker_b: PersonProfile(person_name=speaker_b),
        }

    def _merge_delta(self, profile: PersonProfile, delta: dict) -> None:
        if not isinstance(delta, dict) or not delta:
            return

        # identity: OVERWRITE, whitelist-filtered
        try:
            identity_delta = delta.get("identity") or {}
            if isinstance(identity_delta, dict):
                for k, v in identity_delta.items():
                    if v is None or v == "":
                        continue
                    if k not in PROFILE_IDENTITY_ALLOWED_KEYS:
                        logger.warning("[Profile] identity key dropped (not whitelisted): %s", k)
                        continue
                    if isinstance(v, list):
                        vals = [x for x in v if x is not None and x != ""]
                        if not vals:
                            continue
                        new_value = vals[-1] if isinstance(vals[-1], str) else str(vals[-1])
                    else:
                        new_value = v if isinstance(v, str) else str(v)
                    profile.identity[k] = new_value
            elif identity_delta:
                logger.warning("[Profile] identity delta type invalid: %r", type(identity_delta))
        except Exception as e:
            logger.warning("[Profile] merge identity failed: %s", e)

        # preferences: APPEND + dedup
        try:
            pref_delta = delta.get("preferences") or {}
            if isinstance(pref_delta, dict):
                for sub_key, new_list in pref_delta.items():
                    if new_list is None:
                        continue
                    if isinstance(new_list, str):
                        new_list = [new_list]
                    if not isinstance(new_list, list):
                        continue
                    existing = profile.preferences.setdefault(sub_key, [])
                    seen = {_norm_str(x) for x in existing}
                    for item in new_list:
                        if item is None or item == "":
                            continue
                        norm = _norm_str(item)
                        if norm in seen:
                            continue
                        seen.add(norm)
                        existing.append(item if isinstance(item, str) else str(item))
        except Exception as e:
            logger.warning("[Profile] merge preferences failed: %s", e)

        # traits_and_attitudes: LRU + whitelist
        try:
            ta_delta = delta.get("traits_and_attitudes") or {}
            if isinstance(ta_delta, dict):
                for allowed_k in PROFILE_TA_ALLOWED_KEYS:
                    profile.traits_and_attitudes.setdefault(allowed_k, [])

                for k, new_list in ta_delta.items():
                    k_norm = str(k).strip().lower()
                    if k_norm in PROFILE_TA_ALLOWED_KEYS:
                        target_key = k_norm
                    else:
                        logger.warning("[Profile] traits_and_attitudes non-whitelist key '%s' -> attitudes", k)
                        target_key = "attitudes"

                    if new_list is None:
                        continue
                    if isinstance(new_list, str):
                        new_list = [new_list]
                    if not isinstance(new_list, list):
                        continue

                    existing = profile.traits_and_attitudes[target_key]
                    existing_idx = {_norm_str(x): i for i, x in enumerate(existing)}
                    seen_in_delta: set = set()
                    for item in new_list:
                        if item is None or item == "":
                            continue
                        s_item = item if isinstance(item, str) else str(item)
                        norm = _norm_str(s_item)
                        if not norm or norm in seen_in_delta:
                            continue
                        seen_in_delta.add(norm)
                        if norm in existing_idx:
                            existing[existing_idx[norm]] = None  # tombstone
                        existing.append(s_item)
                    if any(x is None for x in existing):
                        profile.traits_and_attitudes[target_key] = [x for x in existing if x is not None]

                for allowed_k in PROFILE_TA_ALLOWED_KEYS:
                    cur_list = profile.traits_and_attitudes.get(allowed_k, [])
                    if len(cur_list) > PROFILE_TA_MAX_ITEMS_PER_KEY:
                        profile.traits_and_attitudes[allowed_k] = cur_list[-PROFILE_TA_MAX_ITEMS_PER_KEY:]
        except Exception as e:
            logger.warning("[Profile] merge traits_and_attitudes failed: %s", e)

        # relations
        try:
            rels = delta.get("relations") or []
            if isinstance(rels, list):
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue
                    person = rel.get("person") or ""
                    if not person:
                        continue
                    existing_rel = None
                    for r in profile.relations:
                        if r.get("person") == person:
                            existing_rel = r
                            break
                    if existing_rel is None:
                        raw_acts = rel.get("shared_activities") or []
                        if isinstance(raw_acts, str):
                            raw_acts = [raw_acts]
                        clean_acts: list = []
                        seen_new: set = set()
                        for a in (raw_acts if isinstance(raw_acts, list) else []):
                            if a is None or a == "":
                                continue
                            s_a = a if isinstance(a, str) else str(a)
                            n = _norm_str(s_a)
                            if not n or n in seen_new:
                                continue
                            seen_new.add(n)
                            clean_acts.append(s_a)
                        if len(clean_acts) > PROFILE_SHARED_ACTIVITIES_MAX:
                            clean_acts = clean_acts[-PROFILE_SHARED_ACTIVITIES_MAX:]
                        new_rel = {
                            "person": person,
                            "type": rel.get("type", ""),
                            "nickname": rel.get("nickname", ""),
                            "shared_activities": clean_acts,
                        }
                        for k, v in rel.items():
                            if k not in new_rel:
                                new_rel[k] = v
                        profile.relations.append(new_rel)
                    else:
                        new_nick = rel.get("nickname") or ""
                        if new_nick and not existing_rel.get("nickname"):
                            existing_rel["nickname"] = new_nick
                        new_type = rel.get("type") or ""
                        if new_type and not existing_rel.get("type"):
                            existing_rel["type"] = new_type
                        new_acts = rel.get("shared_activities") or []
                        if isinstance(new_acts, str):
                            new_acts = [new_acts]
                        if isinstance(new_acts, list):
                            existing_acts = existing_rel.setdefault("shared_activities", [])
                            existing_idx = {_norm_str(a): i for i, a in enumerate(existing_acts)}
                            seen_in_delta: set = set()
                            for a in new_acts:
                                if a is None or a == "":
                                    continue
                                s_a = a if isinstance(a, str) else str(a)
                                n = _norm_str(s_a)
                                if not n or n in seen_in_delta:
                                    continue
                                seen_in_delta.add(n)
                                if n in existing_idx:
                                    existing_acts[existing_idx[n]] = None
                                existing_acts.append(s_a)
                            if any(x is None for x in existing_acts):
                                existing_acts = [x for x in existing_acts if x is not None]
                                existing_rel["shared_activities"] = existing_acts
                            if len(existing_acts) > PROFILE_SHARED_ACTIVITIES_MAX:
                                existing_rel["shared_activities"] = existing_acts[-PROFILE_SHARED_ACTIVITIES_MAX:]
        except Exception as e:
            logger.warning("[Profile] merge relations failed: %s", e)

        # aggregations
        try:
            agg_delta = delta.get("aggregations") or {}
            if isinstance(agg_delta, dict):
                for k, v in agg_delta.items():
                    try:
                        iv = int(v)
                    except (TypeError, ValueError):
                        continue
                    profile.aggregations[k] = profile.aggregations.get(k, 0) + iv
        except Exception as e:
            logger.warning("[Profile] merge aggregations failed: %s", e)

        # timeline
        try:
            tl_delta = delta.get("timeline") or []
            if isinstance(tl_delta, list):
                seen_pairs = {
                    (_norm_str(e.get("date", "")), _norm_str(e.get("event", "")))
                    for e in profile.timeline
                }
                for ev in tl_delta:
                    if not isinstance(ev, dict):
                        continue
                    event = str(ev.get("event", "")).strip()
                    date = str(ev.get("date", "")).strip()
                    if not event:
                        continue
                    key = (_norm_str(date), _norm_str(event))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    profile.timeline.append(dict(ev))
        except Exception as e:
            logger.warning("[Profile] merge timeline failed: %s", e)

    def _call_llm_extract(
        self,
        speaker_name: str,
        current_profile: PersonProfile,
        chat_current: str,
        chat_history: str,
        current_time: str,
    ) -> dict:
        """Invoke the Venus LLM once for one speaker; return a delta dict."""
        try:
            current_profile_json_str = json.dumps(
                current_profile.to_json(), ensure_ascii=False, indent=2
            )
        except Exception as e:
            logger.warning("[Profile] dump current_profile failed: %s", e)
            current_profile_json_str = "{}"

        try:
            llm_input = PROFILE_EXTRACT_PROMPT.format(
                speaker_name=speaker_name,
                current_profile_json=current_profile_json_str,
                chat_current=chat_current or "",
                chat_history=chat_history or "",
                current_time=current_time or "",
            )
        except Exception as e:
            logger.warning("[Profile] prompt format failed: %s", e)
            return {}

        try:
            raw = venus_chat(
                llm_input,
                model=PERSONA_EXTRACT_MODEL,
                timeout=PERSONA_EXTRACT_TIMEOUT,
                meta={"call_site": "stage7.persona"},
            )
        except Exception as e:
            logger.warning("[Profile] venus_chat failed (speaker=%s): %s", speaker_name, e)
            return {}

        if not raw or not isinstance(raw, str):
            logger.warning("[Profile] LLM empty/non-str output for speaker=%s", speaker_name)
            return {}

        try:
            data = json_repair.loads(raw)
        except Exception as e:
            logger.warning(
                "[Profile] json parse failed (speaker=%s): %s | head=%r",
                speaker_name, e, raw[:500],
            )
            return {}

        if not isinstance(data, dict):
            logger.warning(
                "[Profile] LLM output not a dict (speaker=%s, type=%s)",
                speaker_name, type(data).__name__,
            )
            return {}

        if PROFILE_DEBUG:
            logger.debug(
                "[Profile] delta for %s: %s",
                speaker_name, json.dumps(data, ensure_ascii=False)[:1000],
            )
        return data

    def extract_and_merge_for_window(
        self,
        chat_current: str,
        current_time: str,
        chat_history: str = "",
    ) -> Dict[str, dict]:
        """Run LLM extraction for BOTH speakers on one chat_current window.

        Returns a map {speaker_name -> delta_dict} for debugging / logging.
        The merge is done in place on self.profiles.
        """
        deltas: Dict[str, dict] = {}
        for name, profile in list(self.profiles.items()):
            try:
                delta = self._call_llm_extract(
                    speaker_name=name,
                    current_profile=profile,
                    chat_current=chat_current,
                    chat_history=chat_history,
                    current_time=current_time,
                )
                self._merge_delta(profile, delta)
                deltas[name] = delta
            except Exception as e:
                logger.warning("[Profile] extract_and_merge for %s failed: %s", name, e)
                deltas[name] = {}
        return deltas

    def save_profiles(self, subdir: Optional[str] = None) -> str:
        """Persist current profiles to `<store_root>/<subdir or conv_label>/`.

        Returns the target directory path.
        """
        label = subdir or self.conv_label
        target_dir = os.path.join(self.store_root, label)
        os.makedirs(target_dir, exist_ok=True)
        for name, profile in self.profiles.items():
            fname = _sanitize_filename(name) + ".json"
            fpath = os.path.join(target_dir, fname)
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(profile.to_json(), f, ensure_ascii=False, indent=2)
                logger.info("[Profile] saved to %s", fpath)
            except Exception as e:
                logger.warning("[Profile] save %s failed: %s", fpath, e)
        return target_dir

    def load_profiles(self, subdir: Optional[str] = None) -> None:
        """Reload profiles from disk if they exist (idempotent)."""
        label = subdir or self.conv_label
        target_dir = os.path.join(self.store_root, label)
        if not os.path.isdir(target_dir):
            return
        try:
            files = [f for f in os.listdir(target_dir) if f.endswith(".json")]
        except Exception as e:
            logger.warning("[Profile] list %s failed: %s", target_dir, e)
            return
        for fname in files:
            fpath = os.path.join(target_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = PersonProfile.from_json(data)
                name = profile.person_name or os.path.splitext(fname)[0]
                # Only refresh if the speaker matches one of the two registered.
                if name in self.profiles:
                    self.profiles[name] = profile
                    logger.info("[Profile] loaded %s from %s", name, fpath)
                else:
                    logger.warning("[Profile] skip unknown speaker %s in %s", name, fpath)
            except Exception as e:
                logger.warning("[Profile] read %s failed: %s", fpath, e)

    def render_for_qa(self) -> str:
        blocks: List[str] = []
        # Preserve speaker_a -> speaker_b ordering, regardless of dict order.
        for name in (self.speaker_a, self.speaker_b):
            profile = self.profiles.get(name)
            if profile is None or profile.is_empty():
                continue
            block = self._render_single(name, profile)
            if block:
                blocks.append(block)
        return "\n\n".join(blocks)

    @staticmethod
    def _render_single(speaker_name: str, profile: PersonProfile) -> str:
        lines: List[str] = [f"## Persona Profile — {speaker_name}"]

        # identity
        if profile.identity:
            lines.append("[identity]")
            for k, v in profile.identity.items():
                if not v:
                    continue
                if isinstance(v, list):
                    nonempty = [x for x in v if x]
                    txt = str(nonempty[-1]) if nonempty else ""
                else:
                    txt = str(v)
                if txt:
                    lines.append(f"- {k}: {txt}")

        # traits_and_attitudes
        if profile.traits_and_attitudes and any(v for v in profile.traits_and_attitudes.values()):
            lines.append("[traits_and_attitudes]")
            for k in PROFILE_TA_ALLOWED_KEYS:
                vals = profile.traits_and_attitudes.get(k) or []
                if vals:
                    lines.append(f"- {k}: {', '.join(str(x) for x in vals if x)}")
            for k, vals in profile.traits_and_attitudes.items():
                if k in PROFILE_TA_ALLOWED_KEYS:
                    continue
                if vals:
                    lines.append(f"- {k}: {', '.join(str(x) for x in vals if x)}")

        # preferences
        if profile.preferences and any(v for v in profile.preferences.values()):
            lines.append("[preferences]")
            for sub_key, vals in profile.preferences.items():
                if vals:
                    lines.append(f"- {sub_key}: {', '.join(str(x) for x in vals)}")

        # relations
        if profile.relations:
            rel_lines: List[str] = []
            for r in profile.relations:
                person = r.get("person", "")
                if not person:
                    continue
                rtype = r.get("type", "")
                nick = r.get("nickname", "")
                acts = r.get("shared_activities") or []
                extra = f": {rtype}" if rtype else ""
                line = f"- {person}{extra}"
                suffixes = []
                if nick:
                    suffixes.append(f"nickname: {nick}")
                if acts:
                    suffixes.append(f"shared: {', '.join(str(a) for a in acts)}")
                if suffixes:
                    line += ", " + ", ".join(suffixes)
                rel_lines.append(line)
            if rel_lines:
                lines.append("[relations]")
                lines.extend(rel_lines)

        # aggregations (filtered by MIN_COUNT)
        if profile.aggregations:
            agg_rendered = [
                (k, v) for k, v in profile.aggregations.items()
                if isinstance(v, (int, float)) and v >= PROFILE_AGGREGATION_MIN_COUNT
            ]
            if agg_rendered:
                lines.append("[aggregations]")
                for k, v in agg_rendered:
                    lines.append(f"- {k}: {v}")

        # timeline
        if profile.timeline:
            tl_lines: List[str] = []
            for ev in profile.timeline:
                event = ev.get("event", "")
                if not event:
                    continue
                date = ev.get("date", "")
                date_range = ev.get("date_range", "")
                if date_range:
                    tl_lines.append(f"- {event} (duration: {date_range})")
                elif date:
                    tl_lines.append(f"- {event} ({date})")
                else:
                    tl_lines.append(f"- {event}")
            if tl_lines:
                lines.append("[timeline]")
                lines.extend(tl_lines)

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)
