"""Stage 0 (LoCoMo-Plus): stitch 401 plus samples into 401 per-sample convs.

Each LoCoMo-Plus question carries its own (cue_dialogue, trigger_query, time_gap).
For per-sample memory libraries we must materialise a dedicated conversation
per question -- with the cue dialogue inserted at (last_session_time + 7d -
time_gap) and the trigger_query appended at (last_session_time + 7d).

This stage emits a 401-long JSON list whose every element is a locomo10-style
{"qa": [], "conversation": {speaker_a, speaker_b, session_1, session_1_date_time,
...}} record. Stage1's `load_locomo_raw_data` then iterates len(data) and
produces `scene_list_conv_0..400.json` -- one memory library per plus sample.

Pairing rule: plus_sample[i] uses locomo10[i % len(locomo10)] as its base
conversation. This matches benchmark_eval/locomo_plus/data/unified_input.py
(line: `locomo_item = locomo_list[i % len(locomo_list)]`) byte-for-byte.

Run:
    python -m T_mem.main.stage0_locomo_plus_stitch \
        --locomo10-file  benchmark_eval/locomo/data/locomo10.json \
        --locomo-plus-file benchmark_eval/locomo_plus/data/locomo_plus.json \
        --out-file       data/stitched_locomo_plus.json \
        [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
_T_MEM_ROOT = _HERE.parent.parent.parent
if str(_T_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_T_MEM_ROOT))

# build_conv lives next to the locomo_plus dataset.
_BUILD_CONV_DIR = _T_MEM_ROOT / "benchmark_eval" / "locomo_plus" / "data"
if str(_BUILD_CONV_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_CONV_DIR))

from build_conv import (  # noqa: E402
    parse_ab_dialogue,
    map_speaker,
    parse_time_gap,
    analyze_conversation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage0.locomo_plus_stitch")

# locomo10's session_X_date_time format, e.g. "3:00 PM on 14 March, 2024".
_LOCOMO_TS_FMT = "%I:%M %p on %d %B, %Y"


def _format_locomo_ts(t: datetime) -> str:
    """%I prints zero-padded hour (e.g. '03:00 PM'); locomo10 uses unpadded
    hours (e.g. '3:00 PM'). Strip the leading zero to match the upstream
    format exactly so stage1's `parse_locomo_timestamp` succeeds."""
    s = t.strftime(_LOCOMO_TS_FMT)
    # Replace a leading "0H:" with "H:" if hour is zero-padded.
    if len(s) >= 2 and s[0] == "0" and s[1].isdigit():
        s = s[1:]
    return s


def _build_session_layout_for_sample(
    plus_item: Dict[str, Any],
    locomo_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a locomo10-style `conversation` dict for one plus sample.

    Algorithm (mirrors build_conv.build_context but emits per-session blocks
    instead of a flat dialogue list):

      1. Take all base sessions from `locomo_item['conversation']` verbatim
         (turns + per-session date_time).
      2. Compute cue_time = last_session_time + 7d - parse_time_gap(time_gap)
         and query_time = last_session_time + 7d.
      3. Insert TWO synthetic sessions:
           - cue session at cue_time, containing cue_dialogue's turns
             (A/B remapped to base conversation's speaker_a/speaker_b).
           - query session at query_time, containing trigger_query's turns
             (same A/B remapping).
      4. Sort all (time, turns) pairs by time, then renumber sessions as
         session_1, session_2, ... session_N.
      5. Assign dia_id = "D{session_idx}:{turn_idx}" so stage1 / unified_input
         can index back into individual turns.
    """
    conv = locomo_item["conversation"]
    speaker_a, speaker_b, sessions, session_times = analyze_conversation(conv)

    # cue / query timestamps
    last_t = session_times[-1]
    query_time = last_t + timedelta(days=7)
    back_days = parse_time_gap(plus_item.get("time_gap", "") or "")
    cue_time = query_time - timedelta(days=back_days)

    # remap A/B to real speaker names
    cue_turns = map_speaker(
        parse_ab_dialogue(plus_item.get("cue_dialogue", "") or ""),
        speaker_a,
        speaker_b,
    )
    query_turns = map_speaker(
        parse_ab_dialogue(plus_item.get("trigger_query", "") or ""),
        speaker_a,
        speaker_b,
    )

    # collect (time, turns_list) blocks
    blocks: List[tuple[datetime, List[Dict[str, Any]]]] = []
    for t, sess in zip(session_times, sessions):
        blocks.append((t, copy.deepcopy(sess)))
    if cue_turns:
        blocks.append((cue_time, copy.deepcopy(cue_turns)))
    if query_turns:
        blocks.append((query_time, copy.deepcopy(query_turns)))

    # stable-sort by timestamp
    blocks.sort(key=lambda x: x[0])

    # emit locomo10-style {speaker_a, speaker_b, session_X, session_X_date_time}
    out: Dict[str, Any] = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }
    for i, (t, turns) in enumerate(blocks, start=1):
        # Each turn: {speaker, text, dia_id}. Existing locomo10 turns already
        # carry dia_id; we (re-)assign it consistently across the new layout
        # so downstream `Dn:k` indexing works.
        rebuilt: List[Dict[str, Any]] = []
        for k, raw in enumerate(turns, start=1):
            spk = raw.get("speaker") or "?"
            txt = raw.get("text") or ""
            new_turn: Dict[str, Any] = {
                "speaker": spk,
                "dia_id": f"D{i}:{k}",
                "text": txt,
            }
            # forward optional locomo fields if present
            for opt in ("img_url", "blip_caption", "query"):
                if opt in raw and raw[opt] is not None:
                    new_turn[opt] = raw[opt]
            rebuilt.append(new_turn)
        out[f"session_{i}"] = rebuilt
        out[f"session_{i}_date_time"] = _format_locomo_ts(t)

    return out


def _stitch_one(
    sample_idx: int,
    base_conv_idx: int,
    plus_item: Dict[str, Any],
    locomo_item: Dict[str, Any],
) -> Dict[str, Any]:
    """One 401-list element: locomo10-style record with empty `qa` list."""
    new_conv = _build_session_layout_for_sample(plus_item, locomo_item)
    return {
        "sample_id": f"sample_{sample_idx:03d}",
        "sample_idx": sample_idx,
        "base_conv_idx": base_conv_idx,
        "qa": [],  # plus has no per-conv qa pairs; trigger_query lives at the tail
        "conversation": new_conv,
    }


def stitch_locomo_plus(
    locomo10_file: Path,
    locomo_plus_file: Path,
    out_file: Path,
    limit: int | None = None,
) -> Dict[str, Any]:
    locomo10 = json.load(locomo10_file.open("r", encoding="utf-8"))
    plus = json.load(locomo_plus_file.open("r", encoding="utf-8"))
    n_conv = len(locomo10)
    n_plus_total = len(plus)

    if limit is not None:
        plus = plus[:limit]
    n = len(plus)

    log.info("locomo10=%d  locomo_plus=%d  using=%d", n_conv, n_plus_total, n)

    out_records: List[Dict[str, Any]] = []
    for i, p in enumerate(plus):
        base_idx = i % n_conv
        base = locomo10[base_idx]
        try:
            rec = _stitch_one(i, base_idx, p, base)
        except Exception as e:  # noqa: BLE001
            log.error("[sample_%03d] stitch failed: %s: %s",
                      i, type(e).__name__, e)
            raise
        out_records.append(rec)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %d stitched records -> %s", len(out_records), out_file)

    # Sanity: every stitched conv must be parseable by stage1's loader.
    for rec in out_records[:3]:
        c = rec["conversation"]
        first_t = c["session_1_date_time"]
        try:
            datetime.strptime(first_t, _LOCOMO_TS_FMT)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"timestamp roundtrip failed for {rec['sample_id']}: "
                f"got {first_t!r} ({e})"
            )

    return {
        "n_records": len(out_records),
        "out_file": str(out_file),
        "n_conv_base": n_conv,
        "limit": limit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stitch LoCoMo-Plus 401 samples into 401 locomo10-style convs "
            "(per-sample base via i % len(locomo10))."
        ),
    )
    ap.add_argument(
        "--locomo10-file", required=True,
        help="path to benchmark_eval/locomo/data/locomo10.json",
    )
    ap.add_argument(
        "--locomo-plus-file", required=True,
        help="path to benchmark_eval/locomo_plus/data/locomo_plus.json",
    )
    ap.add_argument(
        "--out-file", required=True,
        help="output stitched JSON path (will be passed to stage1 as T_MEM_DATA_FILE)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="optional: only stitch first N plus samples (for smoke runs).",
    )
    args = ap.parse_args()

    stitch_locomo_plus(
        locomo10_file=Path(args.locomo10_file).resolve(),
        locomo_plus_file=Path(args.locomo_plus_file).resolve(),
        out_file=Path(args.out_file).resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
