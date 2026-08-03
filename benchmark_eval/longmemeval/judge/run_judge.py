"""LongMemEval judge -- 100 % aligned with the official `evaluate_qa.py`.

Source of truth: `LongMemEval-main/src/evaluation/evaluate_qa.py` (5+1 templates,
`'yes' in response.lower()` parsing, abstention detection via `'_abs' in qid`,
no answer-side normalisation, no postprocessing of the hypothesis).

Differences from upstream (deliberate):
  - LLM transport: `VenusLLMProvider` (T-Mem's existing wrapper around the
    Venus gateway) instead of OpenAI's python client, because the rest of
    T-Mem already routes through Venus and the user has confirmed gpt-4o
    is reachable that way.
  - Concurrency: thread-pool of `--concurrency` workers (upstream is purely
    sequential through tqdm). Determinism is preserved because each judge
    call uses temperature=0 and the per-instance label depends only on the
    LLM output for that instance.
  - Output filename `<hyp>.eval-results-<model>` matches upstream byte-for-byte
    so `print_qa_metrics.py` (also upstream) drops in unchanged.
  - We additionally emit a sibling `metrics.json` containing the same numbers
    as `print_qa_metrics.py` would print (Overall / Task-averaged / Abstention
    + 6-qtype breakdown) so downstream tooling does not need to spawn another
    Python invocation.

Inputs:
  --hyp-file  : LongMemEval hypothesis.jsonl  (one {question_id, hypothesis} per line)
  --ref-file  : longmemeval_s_cleaned.json    (the source of qtype/question/answer/qid)

Outputs (in <hyp-file>'s directory unless --out-dir is set):
  <hyp>.eval-results-<judge_model>      -- per-instance jsonl (upstream-compatible)
  <hyp>.eval-results-<judge_model>.metrics.json
                                        -- aggregate metrics (Overall,
                                           Task-averaged, Abstention,
                                           6-qtype breakdown)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.llm.venus_provider import venus_chat  # noqa: E402

# Local import (sibling module).
sys.path.insert(0, str(_HERE.parent))
from judge_prompts import (  # noqa: E402
    SUPPORTED_QTYPES,
    build_judge_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("venus_api_base").setLevel(logging.WARNING)
log = logging.getLogger("lme.judge")

# LongMemEval official judge model. Upstream evaluate_qa.py uses the model
# alias `gpt-4o`; we route through the same alias via the Venus gateway
# (which the user has confirmed routes `gpt-4o` correctly).
DEFAULT_JUDGE_MODEL = "gpt-4o"
DEFAULT_CONCURRENCY = 16
DEFAULT_TIMEOUT = 240
DEFAULT_MAX_RETRIES = 4


def _is_abstention(qid: str) -> bool:
    """Match upstream `'_abs' in entry['question_id']` (note: `in`, not `endswith`)."""
    return "_abs" in (qid or "")


def _judge_one(
    *,
    qtype: str,
    question: str,
    answer: str,
    hypothesis: str,
    abstention: bool,
    judge_model: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """One judge call -> {label: bool, raw: str, latency_s: float}.

    Mirrors the upstream parsing rule:
        completion.choices[0].message.content.strip()
        label = 'yes' in eval_response.lower()
    """
    prompt = build_judge_prompt(
        qtype=qtype,
        question=question,
        answer=answer,
        response=hypothesis,
        abstention=abstention,
    )
    t0 = time.perf_counter()
    try:
        raw = venus_chat(
            prompt,
            model=judge_model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "label":     False,
            "raw":       f"(judge exception: {type(e).__name__}: {e})",
            "latency_s": round(time.perf_counter() - t0, 3),
        }
    raw_str = (raw or "").strip()
    return {
        "label":     "yes" in raw_str.lower(),
        "raw":       raw_str,
        "latency_s": round(time.perf_counter() - t0, 3),
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read jsonl OR json (auto-detect, like upstream evaluate_qa.py)."""
    text = path.read_text(encoding="utf-8")
    text_strip = text.strip()
    if not text_strip:
        return []
    # Try JSON-array first.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fall back to JSONL.
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_reference(ref_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the LongMemEval reference file -> {question_id: full_entry}."""
    data = _load_jsonl(ref_path)
    return {e["question_id"]: e for e in data if "question_id" in e}


def _compute_metrics(
    logs: List[Dict[str, Any]],
    qid2qtype: Dict[str, str],
) -> Dict[str, Any]:
    """Mirror `print_qa_metrics.py`'s breakdown.

    Reports:
      - by-qtype accuracy (over the 6 qtypes that LongMemEval defines)
      - overall accuracy (== upstream "Overall Accuracy")
    """
    type2acc: Dict[str, List[int]] = {qt: [] for qt in SUPPORTED_QTYPES}
    overall: List[int] = []

    for entry in logs:
        qid = entry["question_id"]
        qt = qid2qtype.get(qid)
        ok = bool(entry.get("autoeval_label", {}).get("label"))
        score = 1 if ok else 0
        if qt in type2acc:
            type2acc[qt].append(score)
        overall.append(score)

    by_type: Dict[str, Dict[str, Any]] = {}
    for qt, vs in type2acc.items():
        n = len(vs)
        acc = (sum(vs) / n) if n else 0.0
        by_type[qt] = {"correct": sum(vs), "total": n, "accuracy": round(acc, 4)}

    overall_n = len(overall)
    return {
        "n_judged":                overall_n,
        "by_question_type":        by_type,
        "overall_accuracy":        round((sum(overall) / overall_n) if overall_n else 0.0, 4),
    }


def run_judge(
    hyp_file: Path,
    ref_file: Path,
    out_dir: Path | None,
    judge_model: str,
    concurrency: int,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    if not hyp_file.exists():
        raise SystemExit(f"hypothesis file missing: {hyp_file}")
    if not ref_file.exists():
        raise SystemExit(f"reference file missing: {ref_file}")

    out_dir = out_dir or hyp_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames mirror upstream evaluate_qa.py byte-for-byte
    # (`<hyp_file>.eval-results-<short_model>`), so print_qa_metrics.py
    # would still find them if anyone wants to run the upstream aggregator.
    eval_results_path = out_dir / f"{hyp_file.name}.eval-results-{judge_model}"
    metrics_path = out_dir / f"{hyp_file.name}.eval-results-{judge_model}.metrics.json"

    hypotheses = _load_jsonl(hyp_file)
    qid2qdata = _load_reference(ref_file)
    qid2qtype = {qid: e.get("question_type", "") for qid, e in qid2qdata.items()}

    log.info(
        "loaded hypotheses=%d, reference instances=%d, judge_model=%s, concurrency=%d",
        len(hypotheses), len(qid2qdata), judge_model, concurrency,
    )

    # ------------- judge in parallel -------------
    results: List[Dict[str, Any] | None] = [None] * len(hypotheses)
    n_skipped = 0
    n_done = 0
    lock = threading.Lock()

    def _task(i_entry):
        i, entry = i_entry
        qid = entry.get("question_id", "")
        if qid not in qid2qdata:
            return i, None  # caller filters None as "skipped"
        ref = qid2qdata[qid]
        qtype = ref.get("question_type", "")
        question = ref.get("question", "") or ""
        gold = ref.get("answer", "") or ""
        hyp = entry.get("hypothesis", "") or ""
        abst = _is_abstention(qid)

        judge_out = _judge_one(
            qtype=qtype,
            question=question,
            answer=gold,
            hypothesis=hyp,
            abstention=abst,
            judge_model=judge_model,
            timeout=timeout,
            max_retries=max_retries,
        )
        out_entry = dict(entry)
        out_entry["autoeval_label"] = {
            "model":      judge_model,
            "label":      bool(judge_out["label"]),
            "raw":        judge_out["raw"],
            "latency_s":  judge_out["latency_s"],
            "abstention": bool(abst),
            "qtype":      qtype,
        }
        return i, out_entry

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(_task, (i, e)): i for i, e in enumerate(hypotheses)}
        for fut in as_completed(futs):
            try:
                i, ent = fut.result()
            except Exception as e:  # noqa: BLE001
                i = futs[fut]
                ent = None
                log.warning("judge worker crashed at idx=%d: %s", i, e)
            if ent is None:
                with lock:
                    n_skipped += 1
                continue
            results[i] = ent
            with lock:
                n_done += 1
                if n_done % 50 == 0 or n_done == len(hypotheses):
                    elapsed = time.time() - start
                    log.info(
                        "judged %d/%d (skipped=%d) elapsed=%.0fs",
                        n_done, len(hypotheses), n_skipped, elapsed,
                    )

    logs: List[Dict[str, Any]] = [r for r in results if r is not None]

    # Persist per-instance jsonl (upstream-compatible).
    with eval_results_path.open("w", encoding="utf-8") as f:
        for ent in logs:
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d lines)", eval_results_path, len(logs))

    # Aggregate metrics.
    metrics = _compute_metrics(logs, qid2qtype)
    metrics["judge_model"]   = judge_model
    metrics["hyp_file"]      = str(hyp_file)
    metrics["ref_file"]      = str(ref_file)
    metrics["n_skipped"]     = n_skipped
    metrics["eval_results"]  = str(eval_results_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %s", metrics_path)

    # Pretty-print summary identical to upstream `print_qa_metrics.py` style.
    log.info("=" * 70)
    log.info("LongMemEval judge results  ::  %s", hyp_file.name)
    log.info("=" * 70)
    log.info("By question type:")
    for qt in SUPPORTED_QTYPES:
        v = metrics["by_question_type"].get(qt, {})
        log.info("  %-28s : %.4f  (%d/%d)",
                 qt, v.get("accuracy", 0.0), v.get("correct", 0), v.get("total", 0))
    log.info("-" * 70)
    log.info("  Overall Accuracy       : %.4f  (%d judged)",
             metrics["overall_accuracy"], metrics["n_judged"])
    log.info("=" * 70)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "LongMemEval LLM-as-judge runner -- aligned with the official "
            "evaluate_qa.py + print_qa_metrics.py protocol."
        ),
    )
    ap.add_argument(
        "--hyp-file", type=str, required=True,
        help="LongMemEval hypothesis jsonl produced by stage8_qa_lme.py",
    )
    ap.add_argument(
        "--ref-file", type=str, required=True,
        help="LongMemEval reference json (e.g. longmemeval_s_cleaned.json)",
    )
    ap.add_argument(
        "--out-dir", type=str, default="",
        help="Output directory (default: same directory as --hyp-file)",
    )
    ap.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Judge LLM id (default: {DEFAULT_JUDGE_MODEL})",
    )
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    run_judge(
        hyp_file=Path(args.hyp_file).resolve(),
        ref_file=Path(args.ref_file).resolve(),
        out_dir=out_dir,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
