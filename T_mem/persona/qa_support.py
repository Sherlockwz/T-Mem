#!/usr/bin/env python3
"""Persona-augmentation helpers for the merged-context QA prompt.
Env knobs (all default OFF): T_MEM_MEMORY_TOPK / T_MEM_PERSONA_STORE_ROOT / T_MEM_DUMP_CONTEXTS.
Importing this module has no side effects."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("T_mem.persona.qa_support")


PROFILE_TA_ALLOWED_KEYS: Tuple[str, ...] = ("personality", "values", "attitudes", "beliefs")
PROFILE_AGGREGATION_MIN_COUNT: int = 3


_MEMORY_CELLS_HEADER_RE = re.compile(
    r"^\s*##\s*Relevant\s*Memory\s*Cells\s*:\s*\n", re.MULTILINE
)
_ANY_SECTION_HEADER_RE = re.compile(r"^\s*##\s+", re.MULTILINE)
# CRITICAL: no `\b` after `]` — `]` is non-word so `\b` would silently kill all matches.
_MEMORY_BLOCK_RE = re.compile(r"^\[Memory\s+(\d+)\]", re.MULTILINE)


def truncate_memory_cells_block(ctx: str, top_k: int) -> Tuple[str, Dict[str, Any]]:
    """Keep first top_k [Memory N] blocks under '## Relevant Memory Cells:' (top_k<0 = no-op)."""
    diag: Dict[str, Any] = {
        "section_present": False,
        "n_blocks_before": 0,
        "n_blocks_kept": 0,
        "n_blocks_dropped": 0,
        "top_k_requested": int(top_k),
        "no_op": False,
    }

    if top_k is None or top_k < 0:
        diag["no_op"] = True
        return ctx, diag

    hdr = _MEMORY_CELLS_HEADER_RE.search(ctx)
    if hdr is None:
        diag["no_op"] = True
        return ctx, diag

    diag["section_present"] = True
    section_body_start = hdr.end()

    tail = ctx[section_body_start:]
    next_sec = _ANY_SECTION_HEADER_RE.search(tail)
    if next_sec is None:
        section_body_end = len(ctx)
    else:
        section_body_end = section_body_start + next_sec.start()

    body = ctx[section_body_start:section_body_end]
    matches = list(_MEMORY_BLOCK_RE.finditer(body))
    diag["n_blocks_before"] = len(matches)

    if not matches:
        return ctx, diag

    if top_k >= len(matches):
        diag["n_blocks_kept"] = len(matches)
        return ctx, diag

    keep_end_in_body = matches[top_k].start() if top_k > 0 else 0
    kept_body = body[: keep_end_in_body].rstrip()

    new_ctx = (
        ctx[: section_body_start]
        + kept_body
        + ("\n\n" if kept_body else "\n")
        + ctx[section_body_end:]
    )
    diag["n_blocks_kept"] = top_k
    diag["n_blocks_dropped"] = len(matches) - top_k
    return new_ctx, diag


def _norm_str(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower()


def _downgrade_to_single(raw: Any) -> Dict[str, str]:
    """identity field: dict[str, str] with list-fallback (mirrors ver4.2)."""
    out: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if v is None or v == "":
            continue
        if isinstance(v, list):
            vals = [x for x in v if x is not None and x != ""]
            if not vals:
                continue
            last = vals[-1]
            out[k] = last if isinstance(last, str) else str(last)
        else:
            out[k] = v if isinstance(v, str) else str(v)
    return out


def _coerce_ta(raw: Any) -> Dict[str, List[str]]:
    """traits_and_attitudes: dict[str, list[str]]; merges legacy traits/beliefs_and_attitudes shapes."""
    ta_raw = raw.get("traits_and_attitudes") if isinstance(raw, dict) else None
    if isinstance(ta_raw, dict) and ta_raw:
        out: Dict[str, List[str]] = {}
        for k, v in ta_raw.items():
            if v is None:
                out[k] = []
                continue
            if isinstance(v, list):
                out[k] = [x for x in v if x]
            else:
                out[k] = [v]
        return out

    ta: Dict[str, List[str]] = {k: [] for k in PROFILE_TA_ALLOWED_KEYS}
    if not isinstance(raw, dict):
        return ta
    old_traits = raw.get("traits") or {}
    if isinstance(old_traits, dict):
        for sub_k, sub_v in old_traits.items():
            if not isinstance(sub_v, list):
                sub_v = [sub_v] if sub_v else []
            target = sub_k if sub_k in ("personality", "values") else "personality"
            ta[target].extend(x for x in sub_v if x)
    old_beliefs = raw.get("beliefs_and_attitudes") or {}
    if isinstance(old_beliefs, dict):
        for k, v in old_beliefs.items():
            vals = v if isinstance(v, list) else ([v] if v else [])
            bucket = "attitudes" if ("attitude" in k.lower() or "view_on" in k.lower()) else "beliefs"
            ta[bucket].extend(str(x) for x in vals if x)
    for k in list(ta.keys()):
        seen = set()
        out_list: List[str] = []
        for item in ta[k]:
            s_item = item if isinstance(item, str) else str(item)
            norm = _norm_str(s_item)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out_list.append(s_item)
        ta[k] = out_list
    return ta


def render_persona_md(person_name: str, data: Dict[str, Any]) -> str:
    """Render a profile dict to MD; mirror of ProfileMemory._render_single. Empty -> ''."""
    identity = _downgrade_to_single(data.get("identity"))
    preferences_raw = data.get("preferences") or {}
    preferences: Dict[str, List[str]] = {}
    if isinstance(preferences_raw, dict):
        for k, v in preferences_raw.items():
            if isinstance(v, list):
                preferences[k] = [x for x in v if x]
            elif v:
                preferences[k] = [v]
    ta = _coerce_ta(data)
    relations = data.get("relations") or []
    aggregations = data.get("aggregations") or {}
    timeline = data.get("timeline") or []

    ta_has = any(v for v in ta.values())
    is_empty = not (identity or preferences or ta_has or relations or aggregations or timeline)
    if is_empty:
        return ""

    lines: List[str] = [f"## Persona Profile — {person_name}"]

    if identity:
        lines.append("[identity]")
        for k, v in identity.items():
            if v:
                lines.append(f"- {k}: {v}")

    if ta_has:
        lines.append("[traits_and_attitudes]")
        for k in PROFILE_TA_ALLOWED_KEYS:
            vals = ta.get(k) or []
            if vals:
                lines.append(f"- {k}: {', '.join(str(x) for x in vals if x)}")
        for k, vals in ta.items():
            if k in PROFILE_TA_ALLOWED_KEYS:
                continue
            if vals:
                lines.append(f"- {k}: {', '.join(str(x) for x in vals if x)}")

    if preferences and any(v for v in preferences.values()):
        lines.append("[preferences]")
        for sub_key, vals in preferences.items():
            if vals:
                lines.append(f"- {sub_key}: {', '.join(str(x) for x in vals)}")

    if relations:
        rel_lines: List[str] = []
        for r in relations:
            if not isinstance(r, dict):
                continue
            person = r.get("person", "")
            if not person:
                continue
            rtype = r.get("type", "")
            nick = r.get("nickname", "")
            acts = r.get("shared_activities") or []
            line = f"- {person}"
            if rtype:
                line += f": {rtype}"
            suffixes: List[str] = []
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

    if aggregations:
        agg_rendered = [
            (k, v) for k, v in aggregations.items()
            if isinstance(v, (int, float)) and v >= PROFILE_AGGREGATION_MIN_COUNT
        ]
        if agg_rendered:
            lines.append("[aggregations]")
            for k, v in agg_rendered:
                lines.append(f"- {k}: {v}")

    if timeline:
        tl_lines: List[str] = []
        for ev in timeline:
            if not isinstance(ev, dict):
                continue
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


# CRITICAL: T_mem conv_id is 0-indexed but ver4.2 qa_conv<N> is 1-indexed
# (conv_id=0 ↔ qa_conv1). Multiple timestamp-prefixed dirs may exist; pick lex-last.
def _find_persona_dir(store_root: Path, conv_id: int) -> Optional[Path]:
    if not store_root.is_dir():
        return None
    label = f"qa_conv{conv_id + 1}"
    candidates: List[Path] = []
    for entry in store_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.endswith(f"_{label}"):
            candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def load_persona_block(
    store_root: Optional[Path],
    conv_id: int,
    speaker_hint_a: Optional[str] = None,
    speaker_hint_b: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Load up to two personas, render to MD; ordered by speaker_hint_a/b. Returns (md, diag)."""
    diag: Dict[str, Any] = {
        "store_root_set": store_root is not None,
        "conv_id": conv_id,
        "persona_dir": None,
        "speaker_hints": [speaker_hint_a, speaker_hint_b],
        "files_found": [],
        "speakers_rendered": [],
        "n_blocks": 0,
        "skipped_empty": [],
        "reason_empty": None,
    }

    if store_root is None:
        diag["reason_empty"] = "no_store_root"
        return "", diag

    persona_dir = _find_persona_dir(store_root, conv_id)
    if persona_dir is None:
        diag["reason_empty"] = "no_matching_qa_conv_dir"
        return "", diag
    diag["persona_dir"] = str(persona_dir)

    files = sorted(
        [p for p in persona_dir.iterdir() if p.is_file() and p.suffix == ".json"],
        key=lambda p: p.name,
    )
    diag["files_found"] = [p.name for p in files]
    if not files:
        diag["reason_empty"] = "no_json_files"
        return "", diag

    by_name: Dict[str, Dict[str, Any]] = {}
    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _logger.warning("[persona] load %s failed: %s", p, e)
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("person_name") or p.stem
        by_name[name] = data

    ordered: List[str] = []
    seen: set = set()
    for hint in (speaker_hint_a, speaker_hint_b):
        if hint and hint in by_name and hint not in seen:
            ordered.append(hint)
            seen.add(hint)
    for name in sorted(by_name.keys()):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    blocks: List[str] = []
    for name in ordered:
        md = render_persona_md(name, by_name[name])
        if md:
            blocks.append(md)
            diag["speakers_rendered"].append(name)
        else:
            diag["skipped_empty"].append(name)

    diag["n_blocks"] = len(blocks)
    if not blocks:
        diag["reason_empty"] = "all_profiles_empty_after_render"
        return "", diag

    return "\n\n".join(blocks), diag


_HEADER_SPEAKERS_RE = re.compile(
    r"^\s*Scene memories for conversation between\s+(.+?)\s+and\s+(.+?)\s*:",
    re.IGNORECASE,
)


def parse_speakers_from_ctx(ctx: str) -> Tuple[Optional[str], Optional[str]]:
    """Pull (speaker_a, speaker_b) from baseline ctx header; returns (None,None) on miss."""
    m = _HEADER_SPEAKERS_RE.search(ctx[:500])
    if m is None:
        return None, None
    a = (m.group(1) or "").strip()
    b = (m.group(2) or "").strip()
    return (a or None), (b or None)


def append_persona_section(ctx: str, persona_md: str) -> str:
    """Append persona MD to end of ctx separated by blank line; no-op if empty."""
    if not persona_md or not persona_md.strip():
        return ctx
    left = ctx.rstrip()
    return left + "\n\n" + persona_md.strip() + "\n"


_DUMP_PATH: Optional[Path] = None
_DUMP_LOCK = threading.Lock()


def configure_context_dump(path: Optional[Path]) -> None:
    """Call once from the driver before any context is built; path=None disables dumping."""
    global _DUMP_PATH
    _DUMP_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def dump_context_record(record: Dict[str, Any]) -> None:
    """Append one JSON line to the configured dump path; no-op when disabled."""
    if _DUMP_PATH is None:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _DUMP_LOCK:
        with _DUMP_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_memory_topk_env() -> Optional[int]:
    raw = os.environ.get("T_MEM_MEMORY_TOPK", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        _logger.warning("[persona] ignoring bad T_MEM_MEMORY_TOPK=%r", raw)
        return None
    if v < 0:
        _logger.warning("[persona] ignoring negative T_MEM_MEMORY_TOPK=%d", v)
        return None
    return v


def parse_persona_store_root_env() -> Optional[Path]:
    raw = os.environ.get("T_MEM_PERSONA_STORE_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        _logger.warning("[persona] T_MEM_PERSONA_STORE_ROOT=%s not a directory", p)
        return None
    return p


def parse_dump_contexts_enabled_env() -> bool:
    raw = os.environ.get("T_MEM_DUMP_CONTEXTS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")
