# -*- coding: utf-8 -*-
"""Scene-driven persona extraction scheduler (3-rule R1/R2/R3 algorithm).
R1: buffer >= PERSONA_BUFFER_THRESHOLD turns -> LLM extract + snapshot + clear ("standard").
R3: incoming scene >= PERSONA_BIG_SCENE_THRESHOLD turns -> flush buffer + extract alone ("terminal").
R2: at end-of-conv with non-empty buffer, rollback to last "standard" snapshot + merge tail (Tail-B)."""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from T_mem.persona.common import (
    PERSONA_BUFFER_THRESHOLD,
    PERSONA_BIG_SCENE_THRESHOLD,
)
from T_mem.persona.profile_memory import ProfileMemory, PersonProfile

logger = logging.getLogger(__name__)


def _turns_of(sc: dict) -> int:
    od = sc.get("original_data") or []
    return len(od) if isinstance(od, list) else 0


def _current_time_of(scs: List[dict]) -> str:
    """Reference time for the prompt = timestamp of the last utterance."""
    for sc in reversed(scs):
        od = sc.get("original_data") or []
        if isinstance(od, list):
            for u in reversed(od):
                ts = u.get("original_timestamp") or u.get("timestamp") or ""
                if ts:
                    return str(ts)
    for sc in reversed(scs):
        ts = sc.get("timestamp")
        if ts:
            return str(ts)
    return ""


def _render_chat_current(scs: List[dict]) -> str:
    """Render scenes as '<speaker> [<ts>]: <content>' lines, scenes separated by blank line."""
    segments: List[str] = []
    for sc in scs:
        od = sc.get("original_data") or []
        if not isinstance(od, list):
            continue
        lines: List[str] = []
        for u in od:
            if not isinstance(u, dict):
                continue
            spk = u.get("speaker_name") or u.get("user_name") or u.get("speaker") or "unknown"
            content = u.get("content") or ""
            ts = u.get("original_timestamp") or u.get("timestamp") or ""
            if ts:
                lines.append(f"{spk} [{ts}]: {content}")
            else:
                lines.append(f"{spk}: {content}")
        if lines:
            segments.append("\n".join(lines))
    return "\n\n".join(segments)


@dataclass
class _Chunk:
    """One finished extraction; snapshot_before is set only for kind='standard' (Tail-B rollback target)."""
    kind: str
    scenes: List[dict]
    turns: int
    current_time: str
    snapshot_before: Optional[Dict[str, dict]] = None


class PersonaScheduler:
    """Drives scene-buffered persona extraction for a single conversation."""

    def __init__(
        self,
        profile_mem: ProfileMemory,
        *,
        buffer_threshold: int = PERSONA_BUFFER_THRESHOLD,
        big_scene_threshold: int = PERSONA_BIG_SCENE_THRESHOLD,
        dry_run: bool = False,
    ):
        if buffer_threshold <= 0:
            raise ValueError(f"buffer_threshold must be positive, got {buffer_threshold}")
        if big_scene_threshold <= 0:
            raise ValueError(f"big_scene_threshold must be positive, got {big_scene_threshold}")

        self.profile_mem = profile_mem
        self.buffer_threshold = buffer_threshold
        self.big_scene_threshold = big_scene_threshold
        self.dry_run = dry_run

        self._buffer: List[dict] = []
        self._buffer_turns: int = 0
        self.chunks: List[_Chunk] = []
        self._last_standard_idx: Optional[int] = None

    def run(self, scenes: List[dict]) -> List[_Chunk]:
        """Drive the full 3-rule algorithm; returns the ordered chunk log."""
        cleaned = [sc for sc in (scenes or []) if _turns_of(sc) > 0]
        cleaned.sort(key=lambda sc: str(sc.get("timestamp") or ""))

        for sc in cleaned:
            sc_turns = _turns_of(sc)

            if sc_turns >= self.big_scene_threshold:
                if self._buffer:
                    self._flush_buffer_as_terminal(
                        reason=f"R3_pre_flush (new_sc_turns={sc_turns})"
                    )
                self._extract_chunk(
                    scs=[sc],
                    kind="terminal",
                    reason=f"R3_big_scene (turns={sc_turns})",
                )
                continue

            self._buffer.append(sc)
            self._buffer_turns += sc_turns

            if self._buffer_turns >= self.buffer_threshold:
                self._extract_chunk(
                    scs=list(self._buffer),
                    kind="standard",
                    reason=f"R1_threshold (buffer_turns={self._buffer_turns})",
                )
                self._buffer.clear()
                self._buffer_turns = 0

        self._finalize_tail()

        return list(self.chunks)

    def _snapshot_profiles(self) -> Dict[str, dict]:
        return {name: copy.deepcopy(p.to_json()) for name, p in self.profile_mem.profiles.items()}

    def _restore_profiles(self, snapshot: Dict[str, dict]) -> None:
        if not snapshot:
            return
        for name, data in snapshot.items():
            if name in self.profile_mem.profiles:
                try:
                    self.profile_mem.profiles[name] = PersonProfile.from_json(data)
                except Exception as e:
                    logger.warning(
                        "[PersonaScheduler] restore snapshot for %s failed: %s",
                        name, e,
                    )

    def _extract_chunk(
        self,
        scs: List[dict],
        kind: str,
        reason: str,
    ) -> _Chunk:
        # Standard chunks snapshot BEFORE the LLM call so Tail-B can rollback.
        snapshot_before: Optional[Dict[str, dict]] = None
        if kind == "standard":
            snapshot_before = self._snapshot_profiles()

        current_time = _current_time_of(scs)
        turns = sum(_turns_of(sc) for sc in scs)

        logger.info(
            "[PersonaScheduler] extract chunk kind=%s turns=%d reason=%s current_time=%s",
            kind, turns, reason, current_time,
        )

        if not self.dry_run:
            chat_current = _render_chat_current(scs)
            try:
                self.profile_mem.extract_and_merge_for_window(
                    chat_current=chat_current,
                    current_time=current_time,
                    chat_history="",
                )
            except Exception as e:
                logger.warning(
                    "[PersonaScheduler] LLM extraction failed (kind=%s, reason=%s): %s",
                    kind, reason, e,
                )

        chunk = _Chunk(
            kind=kind,
            scenes=list(scs),
            turns=turns,
            current_time=current_time,
            snapshot_before=snapshot_before,
        )
        self.chunks.append(chunk)
        if kind == "standard":
            self._last_standard_idx = len(self.chunks) - 1
        return chunk

    def _flush_buffer_as_terminal(self, reason: str) -> None:
        # Used by R3 pre-intercept. Does NOT alter _last_standard_idx —
        # rollback anchor must be a `standard` chunk.
        if not self._buffer:
            return
        self._extract_chunk(
            scs=list(self._buffer),
            kind="terminal",
            reason=reason,
        )
        self._buffer.clear()
        self._buffer_turns = 0

    def _finalize_tail(self) -> None:
        """R2 end-of-conversation handling (Tail-B rollback or standalone fallback)."""
        if not self._buffer:
            return

        last_idx = len(self.chunks) - 1
        can_rollback = (
            self._last_standard_idx is not None
            and self._last_standard_idx == last_idx
            and self.chunks[last_idx].kind == "standard"
            and self.chunks[last_idx].snapshot_before is not None
        )

        if can_rollback:
            anchor = self.chunks[self._last_standard_idx]
            if not self.dry_run:
                self._restore_profiles(anchor.snapshot_before or {})

            merged_scs = list(anchor.scenes) + list(self._buffer)
            tail_turns = sum(_turns_of(sc) for sc in self._buffer)

            logger.info(
                "[PersonaScheduler] Tail-B rollback: rollback anchor turns=%d + tail turns=%d => merged turns=%d",
                anchor.turns, tail_turns, anchor.turns + tail_turns,
            )

            # NOTE: keep audit trail — append a replacement chunk rather than popping the anchor.
            self._extract_chunk(
                scs=merged_scs,
                kind="terminal",
                reason=(
                    f"R2_tail_B_rollback (anchor_turns={anchor.turns}, "
                    f"tail_turns={tail_turns})"
                ),
            )
        else:
            logger.info(
                "[PersonaScheduler] Tail-B fallback: buffer standalone (turns=%d, no rollback anchor)",
                self._buffer_turns,
            )
            self._extract_chunk(
                scs=list(self._buffer),
                kind="terminal",
                reason=f"R2_tail_standalone (turns={self._buffer_turns})",
            )

        self._buffer.clear()
        self._buffer_turns = 0


def build_personas_for_conv(
    *,
    conv_id: int,
    scenes: List[dict],
    speaker_a: str,
    speaker_b: str,
    store_root: str,
    conv_label: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[ProfileMemory, List[_Chunk]]:
    """Build personas for one conversation and persist to store_root."""
    label = conv_label or f"qa_conv{conv_id}"
    pm = ProfileMemory(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        store_root=store_root,
        conv_label=label,
    )

    scheduler = PersonaScheduler(pm, dry_run=dry_run)
    chunks = scheduler.run(scenes)

    if not dry_run:
        pm.save_profiles()

    return pm, chunks
