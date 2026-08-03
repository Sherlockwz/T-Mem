#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_hypermem.py — HyperMem-style LLM judge for LongMemEval.

差异对照原有 run_judge.py：
  1. Judge prompt → 单一 CORRECT/WRONG 通用模板（无 qtype 特化模板）
  2. 解析逻辑 → json.loads → "label" == "CORRECT"（而非 "yes" in response）
  3. 模型 → gpt-4o-mini（而非 gpt-4o）
  4. 打分次数 → 每题 3 次，输出 mean ± std
  5. 无 abstention 专门处理（HyperMem 不区分 abstention 题型）

Usage:
  python3 judge_hypermem.py --hyp-file <path> --ref-file <path> [--out-dir <dir>] [--num-runs 3] [--concurrency N]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---- self-contained: ensure T_mem is importable ----
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.llm.venus_provider import venus_chat  # noqa: E402

# ---- local prompt (self-contained) ----
from hypermem_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("venus_api_base").setLevel(logging.WARNING)
log = logging.getLogger("hypermem.judge")

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_NUM_RUNS = 3
DEFAULT_CONCURRENCY = 16
DEFAULT_TIMEOUT = 240
DEFAULT_MAX_RETRIES = 4

# LongMemEval 6 question types (same order as upstream)
SUPPORTED_QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


# ---------------------------------------------------------------------------
# JSON extraction — mirror HyperMem llm_judge.py::_extract_json
# ---------------------------------------------------------------------------

def _extract_json(content: str) -> str:
    """Extract JSON from LLM response that may contain explanation text.

    Handles:
    1. Pure JSON: {"label": "CORRECT"}
    2. Markdown code block: ```json {"label": "CORRECT"} ```
    3. JSON with surrounding text
    """
    # Try 1: Extract from markdown code block
    code_block_match = re.search(
        r'```(?:json)?\s*(\{[^`]*\})\s*```', content, re.DOTALL
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try 2: Find JSON object with "label" key
    json_match = re.search(
        r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', content
    )
    if json_match:
        return json_match.group(0)

    # Try 3: Return original content (let json.loads handle it)
    return content.strip()


# ---------------------------------------------------------------------------
# Single judge call — HyperMem style
# ---------------------------------------------------------------------------

def _judge_one_hypermem(
    *,
    question: str,
    golden_answer: str,
    generated_answer: str,
    judge_model: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """One judge call → {label: bool, raw: str, latency_s: float}.

    HyperMem protocol: JSON output with "label": "CORRECT" / "WRONG".
    """
    user_prompt = JUDGE_USER_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

    t0 = time.perf_counter()
    try:
        raw = venus_chat(
            full_prompt,
            model=judge_model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=0.0,
        )
    except Exception as e:
        return {
            "label":     False,
            "raw":       f"(judge exception: {type(e).__name__}: {e})",
            "latency_s": round(time.perf_counter() - t0, 3),
        }

    raw_str = (raw or "").strip()
    label = False

    if raw_str:
        json_str = _extract_json(raw_str)
        if json_str:
            try:
                result = json.loads(json_str)
                lbl = (result.get("label") or "").strip().upper()
                label = (lbl == "CORRECT")
            except (json.JSONDecodeError, AttributeError):
                pass

    return {
        "label":     label,
        "raw":       raw_str,
        "latency_s": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read jsonl or json (auto-detect)."""
    text = path.read_text(encoding="utf-8")
    text_strip = text.strip()
    if not text_strip:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_reference(ref_path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_jsonl(ref_path)
    return {e["question_id"]: e for e in data if "question_id" in e}


# ---------------------------------------------------------------------------
# Metrics computation — HyperMem style (3-run mean ± std)
# ---------------------------------------------------------------------------

def _compute_metrics_hypermem(
    logs: List[Dict[str, Any]],
    qid2qtype: Dict[str, str],
    num_runs: int,
) -> Dict[str, Any]:
    """Compute metrics with multi-run mean ± std, per LongMemEval 6 qtypes.

    Unlike original run_judge.py which does single-run yes/no, this computes
    per-run accuracy then mean±std across runs, matching HyperMem protocol.
    """
    # Collect per-run per-qtype scores
    type_correct: Dict[str, List[int]] = {qt: [0] * num_runs for qt in SUPPORTED_QTYPES}
    type_total: Dict[str, int] = {qt: 0 for qt in SUPPORTED_QTYPES}

    run_scores: List[float] = []
    for run_idx in range(num_runs):
        correct_count = 0
        total_count = 0
        for entry in logs:
            qid = entry["question_id"]
            qt = qid2qtype.get(qid)
            judgments = entry.get("llm_judgments", {})
            jkey = f"judgment_{run_idx + 1}"
            ok = bool(judgments.get(jkey, False))
            if run_idx == 0 and qt in type_total:
                type_total[qt] += 1
            if qt in type_correct and ok:
                type_correct[qt][run_idx] += 1
            if ok:
                correct_count += 1
            total_count += 1
        if total_count > 0:
            run_scores.append(correct_count / total_count)

    mean_acc = float(np.mean(run_scores)) if run_scores else 0.0
    std_acc = float(np.std(run_scores)) if run_scores else 0.0

    by_type: Dict[str, Dict[str, Any]] = {}
    for qt in SUPPORTED_QTYPES:
        n = type_total[qt]
        if n > 0:
            cat_accs = [type_correct[qt][i] / n for i in range(num_runs)]
            by_type[qt] = {
                "correct": type_correct[qt],
                "total": n,
                "mean": round(float(np.mean(cat_accs)), 4),
                "std": round(float(np.std(cat_accs)), 4),
                "individual_runs": [round(a, 4) for a in cat_accs],
            }
        else:
            by_type[qt] = {
                "correct": [0] * num_runs,
                "total": 0,
                "mean": 0.0,
                "std": 0.0,
                "individual_runs": [0.0] * num_runs,
            }

    return {
        "overall": {
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "run_scores": [round(s, 4) for s in run_scores],
        },
        "by_question_type": by_type,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_judge_hypermem(
    hyp_file: Path,
    ref_file: Path,
    out_dir: Path | None,
    judge_model: str,
    num_runs: int,
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

    eval_results_path = out_dir / f"{hyp_file.name}.eval-results-{judge_model}"
    metrics_path = out_dir / f"{hyp_file.name}.eval-results-{judge_model}.metrics.json"

    hypotheses = _load_jsonl(hyp_file)
    qid2qdata = _load_reference(ref_file)
    qid2qtype = {qid: e.get("question_type", "") for qid, e in qid2qdata.items()}

    log.info(
        "loaded hypotheses=%d, reference instances=%d, judge_model=%s, num_runs=%d, concurrency=%d",
        len(hypotheses), len(qid2qdata), judge_model, num_runs, concurrency,
    )

    n_judge_calls = len(hypotheses) * num_runs
    log.info("total judge calls = %d (%d hypotheses × %d runs)", n_judge_calls, len(hypotheses), num_runs)

    # ---- Judge all hypotheses × runs in parallel ----
    results: List[Dict[str, Any] | None] = [None] * len(hypotheses)
    n_skipped = 0
    n_done = 0
    lock = threading.Lock()

    def _task(i_entry_and_run):
        i, entry, run_idx = i_entry_and_run
        qid = entry.get("question_id", "")
        if qid not in qid2qdata:
            return i, run_idx, None
        ref = qid2qdata[qid]
        question = ref.get("question", "") or ""
        gold = ref.get("answer", "") or ""
        hyp = entry.get("hypothesis", "") or ""

        judge_out = _judge_one_hypermem(
            question=question,
            golden_answer=gold,
            generated_answer=hyp,
            judge_model=judge_model,
            timeout=timeout,
            max_retries=max_retries,
        )
        return i, run_idx, judge_out

    start = time.time()

    # Flatten: create (i, entry, run_idx) for each hypothesis × run
    all_work: List[Tuple[int, Dict, int]] = []
    for i, entry in enumerate(hypotheses):
        for r in range(num_runs):
            all_work.append((i, entry, r))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(_task, w): w for w in all_work}
        for fut in as_completed(futs):
            try:
                i, run_idx, judge_out = fut.result()
            except Exception as e:
                log.warning("judge worker crashed: %s", e)
                continue
            if judge_out is None:
                with lock:
                    n_skipped += 1
                continue

            with lock:
                if results[i] is None:
                    results[i] = dict(hypotheses[i])
                    results[i]["llm_judgments"] = {}
                results[i]["llm_judgments"][f"judgment_{run_idx + 1}"] = bool(judge_out["label"])

                # Store raw responses (first run wins for display)
                if "judge_raw" not in results[i]:
                    results[i]["judge_raw"] = {}
                results[i]["judge_raw"][f"run_{run_idx + 1}"] = judge_out["raw"]

                n_done += 1
                if n_done % 100 == 0 or n_done == n_judge_calls:
                    elapsed = time.time() - start
                    log.info(
                        "judged %d/%d calls (skipped=%d) elapsed=%.0fs",
                        n_done, n_judge_calls, n_skipped, elapsed,
                    )

    logs: List[Dict[str, Any]] = [r for r in results if r is not None]

    # Persist per-instance jsonl
    with eval_results_path.open("w", encoding="utf-8") as f:
        for ent in logs:
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d lines)", eval_results_path, len(logs))

    # Aggregate metrics (HyperMem style)
    metrics = _compute_metrics_hypermem(logs, qid2qtype, num_runs)
    metrics["judge_model"] = judge_model
    metrics["num_runs"] = num_runs
    metrics["hyp_file"] = str(hyp_file)
    metrics["ref_file"] = str(ref_file)
    metrics["n_skipped"] = n_skipped
    metrics["n_judged"] = len(logs)
    metrics["eval_results"] = str(eval_results_path)

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %s", metrics_path)

    # Pretty-print summary
    log.info("=" * 70)
    log.info("HyperMem judge results  ::  %s  (model=%s, runs=%d)",
             hyp_file.name, judge_model, num_runs)
    log.info("=" * 70)
    log.info("By question type:")
    for qt in SUPPORTED_QTYPES:
        v = metrics["by_question_type"].get(qt, {})
        if v.get("total", 0) > 0:
            log.info("  %-28s : %.4f ± %.4f  (%d total)",
                     qt, v.get("mean", 0.0), v.get("std", 0.0), v.get("total", 0))
        else:
            log.info("  %-28s : (no data)", qt)
    log.info("-" * 70)
    overall = metrics["overall"]
    log.info("  Overall Accuracy       : %.4f ± %.4f",
             overall["mean_accuracy"], overall["std_accuracy"])
    log.info("  Run scores             : %s", overall["run_scores"])
    log.info("=" * 70)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HyperMem-style LongMemEval judge — single CORRECT/WRONG template + 3-run mean±std",
    )
    ap.add_argument(
        "--hyp-file", type=str, required=True,
        help="LongMemEval hypothesis jsonl (e.g. hypothesis_hypermem.jsonl)",
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
    ap.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                    help=f"Number of judge runs per question (default: {DEFAULT_NUM_RUNS})")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    run_judge_hypermem(
        hyp_file=Path(args.hyp_file).resolve(),
        ref_file=Path(args.ref_file).resolve(),
        out_dir=out_dir,
        judge_model=args.judge_model,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
