"""Stage 0 (LongMemEval): convert LongMemEval-S into 500 per-instance convs.

LongMemEval ships 500 self-contained question instances; each instance carries
its OWN haystack history (38--62 sessions, 396--616 turns, ~115k tokens). To
reuse the T-Mem build pipeline (stages 1-7) verbatim, we materialise each
instance as its own locomo10-style conversation. Stage1's `load_locomo_raw_data`
then iterates `len(data)` and produces `scene_list_conv_0..N-1.json` -- one
memory library per LongMemEval question.

Key format conversions (LongMemEval -> locomo10):
  * timestamp: `"2023/05/20 (Sat) 02:21"` -> `"2:21 AM on 20 May, 2023"`
    (locomo10 timestamp accepted by stage1.parse_locomo_timestamp).
  * roles `user`/`assistant` are emitted as fixed speaker names `"User"`/
    `"Assistant"`. Stage1's `speaker_name_to_id` auto-appends `_{conv_id}`,
    so 500 conversations never collide on speaker_id.
  * `dia_id = "D{session_idx}:{turn_idx}"` (1-indexed, matches LoCoMo-Plus).
  * `has_answer` flags on individual turns are dropped (T-Mem's build does
    not consume them; the judge reads them from the original LongMemEval
    file when needed).

Per-instance `qa[]` field (consumed by stage5/stage6 to drive retrieval):
  * Exactly one QA entry per instance: `{question, answer, category, evidence}`.
  * `category = 1` (any value != 5 works; stage5/stage6 hard-skip category==5).

Top-level extras kept on each stitched record (used by stage8_qa_lme; stage1
ignores anything outside `conversation` and `qa`):
  * `lme_question_id`, `lme_question_type`, `lme_question_date`,
    `lme_answer`, `lme_haystack_session_ids`, `lme_answer_session_ids`.

Run:
    python -m T_mem.main.stage0_lme_stitch \
        --lme-file       benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json \
        --out-file       <experiment_dir>/data/stitched_lme.json \
        [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
_T_MEM_ROOT = _HERE.parent.parent.parent
if str(_T_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_T_MEM_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage0.lme_stitch")

# locomo10 / stage1.parse_locomo_timestamp expects e.g. "3:00 PM on 14 March, 2024".
_LOCOMO_TS_FMT = "%I:%M %p on %d %B, %Y"

# LongMemEval timestamp: "2023/05/20 (Sat) 02:21" -- strip the weekday parenthetical.
_LME_WEEKDAY_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")
_LME_TS_FMT = "%Y/%m/%d %H:%M"

# Fixed speaker names. stage1 auto-disambiguates via `speaker_name_to_id =
# f"{name.lower().replace(' ', '_')}_{con_id}"`, so all 500 conversations
# get distinct speaker_ids ("user_0", "user_1", ..., "user_499").
SPEAKER_USER = "User"
SPEAKER_ASSISTANT = "Assistant"

# stage5_retrieval_locomo / stage6 hard-skip qa entries with `category == 5`
# (LoCoMo's "open-ended" bucket); any other value works, we use 1 as a
# neutral placeholder.
_LME_QA_CATEGORY = 1


def _parse_lme_timestamp(ts: str) -> datetime:
    """LongMemEval `"2023/05/20 (Sat) 02:21"` -> `datetime`."""
    cleaned = _LME_WEEKDAY_RE.sub(" ", ts).strip()
    return datetime.strptime(cleaned, _LME_TS_FMT)


def _format_locomo_ts(t: datetime) -> str:
    """`datetime` -> locomo10 timestamp like `"2:21 AM on 20 May, 2023"`.

    `%I` zero-pads the hour ("02:21 AM"); locomo10 uses an unpadded hour
    ("2:21 AM"), and stage1.parse_locomo_timestamp matches the unpadded form.
    Strip a single leading zero to align byte-for-byte with locomo10.
    """
    s = t.strftime(_LOCOMO_TS_FMT)
    if len(s) >= 2 and s[0] == "0" and s[1].isdigit():
        s = s[1:]
    return s


def _build_session_layout(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one LongMemEval instance's haystack into a locomo10 `conversation` dict."""
    dates = entry.get("haystack_dates") or []
    sessions = entry.get("haystack_sessions") or []
    if len(dates) != len(sessions):
        raise ValueError(
            f"haystack_dates/haystack_sessions length mismatch for {entry.get('question_id')!r}: "
            f"{len(dates)} vs {len(sessions)}"
        )

    # Sort sessions by parsed timestamp. longmemeval_s_cleaned.json already
    # sorts these, but the oracle release does not -- sort defensively so
    # the same code works for both.
    indexed = []
    for i, (d, s) in enumerate(zip(dates, sessions)):
        try:
            ts = _parse_lme_timestamp(d)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"unparseable haystack_dates[{i}]={d!r} for {entry.get('question_id')!r}: {e}"
            )
        indexed.append((ts, s))
    indexed.sort(key=lambda x: x[0])

    out: Dict[str, Any] = {
        "speaker_a": SPEAKER_USER,
        "speaker_b": SPEAKER_ASSISTANT,
    }
    for sess_idx, (sess_t, turns) in enumerate(indexed, start=1):
        rebuilt: List[Dict[str, Any]] = []
        for turn_idx, raw in enumerate(turns, start=1):
            if not isinstance(raw, dict):
                # Defensive: malformed turn; skip and warn.
                log.warning(
                    "[%s] session_%d turn_%d is %s, expected dict; skipping",
                    entry.get("question_id"), sess_idx, turn_idx,
                    type(raw).__name__,
                )
                continue
            role = _coerce_str(raw.get("role")).strip().lower()
            content = _coerce_str(raw.get("content")).strip()
            if not content:
                # Skip empty turns; stage1's scene extractor cannot consume them.
                continue
            speaker = SPEAKER_USER if role == "user" else SPEAKER_ASSISTANT
            rebuilt.append({
                "speaker": speaker,
                "dia_id": f"D{sess_idx}:{turn_idx}",
                "text": content,
            })
        if not rebuilt:
            # Session collapses to nothing after empty-turn pruning -- emit
            # an empty list anyway so session indexing stays contiguous.
            log.warning(
                "[%s] session_%d collapsed to 0 turns after empty filtering",
                entry.get("question_id"), sess_idx,
            )
        out[f"session_{sess_idx}"] = rebuilt
        out[f"session_{sess_idx}_date_time"] = _format_locomo_ts(sess_t)

    return out


def _coerce_str(v: Any) -> str:
    """LongMemEval `answer` is sometimes a non-string (int / float, e.g. '18' for
    a temporal-reasoning day count). Coerce to str so downstream `.strip()` /
    JSON serialisation never crashes; preserves None as ''.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _stitch_one(idx: int, entry: Dict[str, Any]) -> Dict[str, Any]:
    """One stitched record: locomo10-style + LongMemEval extras + 1-entry qa."""
    question = _coerce_str(entry.get("question")).strip()
    answer = _coerce_str(entry.get("answer")).strip()
    qid = _coerce_str(entry.get("question_id")).strip() or f"lme_{idx:04d}"

    if not question:
        raise ValueError(f"empty question for instance idx={idx} qid={qid!r}")

    conv = _build_session_layout(entry)

    return {
        # T-Mem-side fields consumed by stage1..stage7.
        "sample_id": f"lme_{idx:04d}",
        "sample_idx": idx,
        "qa": [{
            "question": question,
            "answer": answer,
            "category": _LME_QA_CATEGORY,
            "evidence": [],
            # `adversarial_answer` is occasionally read by qa-eval scripts
            # in LoCoMo; safe to omit -- stage5/6/8 do not require it.
        }],
        "conversation": conv,
        # LongMemEval-side fields consumed by stage8_qa_lme + judge.
        # stage1 ignores anything outside `qa` / `conversation`.
        "lme_question_id":          qid,
        "lme_question_type":        _coerce_str(entry.get("question_type")),
        "lme_question_date":        _coerce_str(entry.get("question_date")),
        "lme_answer":               answer,
        "lme_haystack_session_ids": list(entry.get("haystack_session_ids") or []),
        "lme_answer_session_ids":   list(entry.get("answer_session_ids") or []),
    }


def stitch_lme(
    lme_file: Path,
    out_file: Path,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Convert LongMemEval-S into N stitched locomo10-style records (one per question)."""
    data = json.loads(lme_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(
            f"unexpected LME root type {type(data).__name__}; expected list"
        )
    n_total = len(data)

    if limit is not None and limit > 0:
        data = data[:limit]
    n = len(data)

    log.info("lme_total=%d  using=%d  out=%s", n_total, n, out_file)

    out_records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for i, entry in enumerate(data):
        try:
            rec = _stitch_one(i, entry)
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            qid = entry.get("question_id") if isinstance(entry, dict) else "?"
            log.error("[idx=%d qid=%s] stitch failed: %s: %s\n%s",
                      i, qid, type(e).__name__, e, _tb.format_exc())
            skipped.append({"idx": i, "question_id": qid, "error": str(e)})
            continue
        out_records.append(rec)

    if not out_records:
        raise SystemExit("stitch produced 0 records; aborting")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %d stitched records -> %s (skipped=%d)",
             len(out_records), out_file, len(skipped))

    # Sanity round-trip: every stitched conv must be parseable by stage1's
    # `parse_locomo_timestamp` for at least its session_1 timestamp.
    # We probe the first 3 records; failures abort hard so we surface them
    # before the build pipeline burns LLM credits on broken data.
    for rec in out_records[:3]:
        c = rec["conversation"]
        try:
            first_t = c["session_1_date_time"]
            datetime.strptime(first_t, _LOCOMO_TS_FMT)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"timestamp roundtrip failed for {rec['sample_id']}: "
                f"got {c.get('session_1_date_time')!r} ({e})"
            )

    return {
        "n_records":  len(out_records),
        "n_skipped":  len(skipped),
        "skipped":    skipped,
        "out_file":   str(out_file),
        "lme_total":  n_total,
        "limit":      limit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert LongMemEval-S into N locomo10-style conversations "
            "(one per question instance) for the T-Mem build pipeline."
        ),
    )
    ap.add_argument(
        "--lme-file", required=True,
        help="path to longmemeval_s_cleaned.json (or a smoke-subset file with the same schema)",
    )
    ap.add_argument(
        "--out-file", required=True,
        help="output stitched JSON path (will be passed to stage1 as T_MEM_DATA_FILE)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="optional: only stitch first N instances (for smoke runs)",
    )
    args = ap.parse_args()

    stitch_lme(
        lme_file=Path(args.lme_file).resolve(),
        out_file=Path(args.out_file).resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
