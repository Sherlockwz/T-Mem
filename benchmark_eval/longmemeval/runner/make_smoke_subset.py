"""Build a small balanced smoke subset of LongMemEval-S for CI / pipeline checks.

Samples K instances per question-type plus K abstention instances (the 30
`_abs`-suffixed instances span the 6 main types, but we want them sampled
explicitly so the abstention judge prompt is exercised). The default
configuration mirrors the plan agreed with the user:

    - 2 single-session-user (non-abs)
    - 2 single-session-assistant (non-abs)
    - 2 single-session-preference (non-abs)
    - 2 multi-session (non-abs)
    - 2 temporal-reasoning (non-abs)
    - 2 knowledge-update (non-abs)
    - 2 abstention (sampled across types)
    -------------------------------------
    14 instances total

The output JSON has exactly the same schema as `longmemeval_s_cleaned.json`,
so it drops in as a `--lme-file` to stage0_lme_stitch.

Run:
    python -m benchmark_eval.longmemeval.runner.make_smoke_subset \
        --lme-file benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json \
        --out-file benchmark_eval/longmemeval/data/longmemeval_s_smoke14.json \
        [--per-type 2] [--abs-count 2] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lme.make_smoke_subset")

# Six question-type buckets used by LongMemEval. Sampling is per non-abstention
# bucket; abstention is sampled separately because it cuts across all types.
QTYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def _is_abstention(entry: Dict[str, Any]) -> bool:
    return str(entry.get("question_id", "")).endswith("_abs")


def make_smoke_subset(
    lme_file: Path,
    out_file: Path,
    per_type: int = 2,
    abs_count: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    data = json.loads(lme_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"unexpected LME root type: {type(data).__name__}")

    rng = random.Random(seed)

    # Bucket: non-abstention by qtype, plus one abstention bucket.
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    abs_bucket: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if _is_abstention(entry):
            abs_bucket.append(entry)
            continue
        qt = entry.get("question_type")
        if qt in QTYPES:
            by_type[qt].append(entry)

    # Deterministic sample: shuffle each bucket with the given seed, take
    # head-K. Using shuffle (rather than rng.sample) keeps the total order
    # of returned instances stable across reruns when buckets shrink.
    selected: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "lme_file": str(lme_file),
        "seed": seed,
        "per_type": per_type,
        "abs_count": abs_count,
        "by_type": {},
        "abstention": {},
    }

    for qt in QTYPES:
        bucket = list(by_type.get(qt, []))
        rng.shuffle(bucket)
        chosen = bucket[:per_type]
        selected.extend(chosen)
        summary["by_type"][qt] = {
            "available": len(by_type.get(qt, [])),
            "chosen":    len(chosen),
            "ids":       [c["question_id"] for c in chosen],
        }
        log.info("[%s] available=%d  chosen=%d", qt, len(bucket), len(chosen))

    # Abstention: deduplicate against already-selected ids (an `_abs` id
    # technically can't collide with a non-abs id, but be defensive).
    chosen_ids = {c["question_id"] for c in selected}
    abs_pool = [e for e in abs_bucket if e["question_id"] not in chosen_ids]
    rng.shuffle(abs_pool)
    abs_chosen = abs_pool[:abs_count]
    selected.extend(abs_chosen)
    summary["abstention"] = {
        "available": len(abs_bucket),
        "chosen":    len(abs_chosen),
        "ids":       [c["question_id"] for c in abs_chosen],
    }
    log.info("[abstention] available=%d  chosen=%d",
             len(abs_bucket), len(abs_chosen))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %d smoke instances -> %s", len(selected), out_file)

    summary_path = out_file.with_suffix(out_file.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote summary -> %s", summary_path)

    summary["n_selected"] = len(selected)
    summary["out_file"] = str(out_file)
    summary["summary_file"] = str(summary_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sample a small balanced smoke subset (default 14 instances) "
            "from LongMemEval-S for CI / pipeline checks."
        ),
    )
    ap.add_argument(
        "--lme-file", required=True,
        help="path to longmemeval_s_cleaned.json",
    )
    ap.add_argument(
        "--out-file", required=True,
        help="output JSON path (same schema as the input file)",
    )
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--abs-count", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    make_smoke_subset(
        lme_file=Path(args.lme_file).resolve(),
        out_file=Path(args.out_file).resolve(),
        per_type=args.per_type,
        abs_count=args.abs_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
