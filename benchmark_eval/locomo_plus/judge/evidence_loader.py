"""Build sample_id -> (evidence, relation_type) maps for Locomo-Plus judging.
Evidence is locomo_plus.json[sample_idx]['cue_dialogue'] with A:/B: replaced by real speaker names."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .judge_prompts import replace_ab_with_names
except ImportError:
    from judge_prompts import replace_ab_with_names  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
LOCOMO_PLUS_DIR = HERE.parent

DEFAULT_LOCOMO_PLUS_FILE = LOCOMO_PLUS_DIR / "data" / "locomo_plus.json"


def load_locomo_plus(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path is not None else DEFAULT_LOCOMO_PLUS_FILE
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"locomo_plus.json at {p} is not a list")
    return data


def sample_idx_from_id(sample_id: str) -> int:
    """Parse 'sample_xxx' into integer index (e.g. 'sample_042' -> 42)."""
    try:
        return int(sample_id.split("_", 1)[1])
    except Exception as e:
        raise ValueError(f"invalid sample_id={sample_id!r}") from e


def build_evidence_and_reltype_maps(
    predictions: list[dict[str, Any]],
    locomo_plus: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (evidence_map, relation_type_map) keyed by sample_id."""
    if locomo_plus is None:
        locomo_plus = load_locomo_plus()

    n_lp = len(locomo_plus)
    evidence_map: dict[str, str] = {}
    rt_map: dict[str, str] = {}

    for rec in predictions:
        sid = rec.get("sample_id", "")
        if not sid:
            continue
        sidx = sample_idx_from_id(sid)
        if not (0 <= sidx < n_lp):
            raise IndexError(
                f"sample_idx={sidx} (from sample_id={sid!r}) out of "
                f"locomo_plus range [0,{n_lp})"
            )
        lp_item = locomo_plus[sidx]
        raw_cue = lp_item.get("cue_dialogue", "") or ""
        sa = rec.get("speaker_a", "A") or "A"
        sb = rec.get("speaker_b", "B") or "B"
        evidence_map[sid] = replace_ab_with_names(raw_cue, sa, sb)
        rt_map[sid] = lp_item.get("relation_type", "unknown") or "unknown"

    return evidence_map, rt_map
