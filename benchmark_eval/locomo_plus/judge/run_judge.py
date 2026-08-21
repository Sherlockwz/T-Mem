"""LLM-as-judge evaluator for Locomo-Plus predictions.
Input: predictions JSONL with (question, gold, pred, sample_idx); judge via an OpenAI-compatible LLM.
Output: judged JSONL with per-item {label, reason} + aggregate accuracy (majority vote)."""
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
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import json_repair
except ImportError:
    json_repair = None

HERE = Path(__file__).resolve().parent
LOCOMO_PLUS_DIR = HERE.parent
PROJECT_ROOT = LOCOMO_PLUS_DIR.parent.parent
LOCOMO_RUNNER_DIR = PROJECT_ROOT / "benchmark_eval" / "locomo" / "runner"

for p in (HERE, LOCOMO_RUNNER_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from judge_prompts import PROMPT_REGISTRY, get_prompt  # noqa: E402
from evidence_loader import build_evidence_and_reltype_maps  # noqa: E402
from memos_judge import call_llm  # noqa: E402
from T_mem.config import MODELS  # noqa: E402


DEFAULT_JUDGE_MODEL = MODELS["locomo_plus_judge"]
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 4
DEFAULT_CONCURRENCY = 16
DEFAULT_NUM_RUNS = 1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("root").setLevel(logging.WARNING)
log = logging.getLogger("locomo_plus.judge")


def _parse_judge_output(response: str) -> dict[str, Any]:
    """Parse {label, reason} from the judge LLM response; any parse failure -> 'wrong'."""
    if not response or response == "error":
        return {"label": "wrong", "reason": "judge LLM error"}

    if json_repair is not None:
        try:
            obj = json_repair.loads(response)
            if isinstance(obj, str):
                obj = json_repair.loads(obj)
            if isinstance(obj, dict) and "label" in obj:
                lab = str(obj["label"]).strip().lower()
                if lab not in ("correct", "wrong"):
                    lab = "wrong"
                return {"label": lab, "reason": obj.get("reason", "")}
        except Exception:
            pass

    lab_m = re.search(r'"label"\s*:\s*"(correct|wrong)"', response, re.IGNORECASE)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', response)
    lab = (lab_m.group(1).lower() if lab_m else "wrong")
    return {"label": lab, "reason": (reason_m.group(1) if reason_m else "")}


def judge_one(
    question: str,
    evidence: str,
    pred: str,
    prompt_template: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    """Judge a single (question, evidence, pred) triple."""
    if not pred or pred == "error" or "(Error" in pred or "(Worker crash" in pred:
        return {"label": "wrong", "reason": "prediction was an error"}

    prompt = prompt_template.format(
        question=question,
        evidence=evidence,
        pred=pred,
    )
    # call_llm() has a fixed 240s timeout and 3-retry loop baked in
    # (see benchmark_eval/locomo/runner/memos_judge.py::call_llm), so the
    # `timeout` / `max_retries` kwargs accepted by judge_one are intentionally
    # not forwarded here. They are kept in the signature for CLI / API
    # compatibility with the original run_judge entry points.
    _ = (timeout, max_retries)
    t0 = time.perf_counter()
    try:
        resp = call_llm(
            prompt,
            model=model,
        )
    except Exception as e:  # pragma: no cover - defensive
        return {"label": "wrong", "reason": f"judge exception: {e}"}
    parsed = _parse_judge_output(resp)
    parsed["latency_s"] = round(time.perf_counter() - t0, 3)
    return parsed


def _write_metrics(
    out_dir: Path,
    judged: dict[str, Any],
    rt_map: dict[str, str],
) -> dict[str, Any]:
    details = judged["details"]
    total = len(details)
    n_correct = sum(1 for d in details if d["final_label"] == "correct")
    acc = n_correct / total if total else 0.0

    by_rt: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in details:
        rt = d.get("relation_type") or "unknown"
        by_rt[rt]["total"] += 1
        if d["final_label"] == "correct":
            by_rt[rt]["correct"] += 1

    metrics = {
        "method": judged["method"],
        "prompt": judged["prompt"],
        "judge_model": judged["judge_model"],
        "num_runs": judged["num_runs"],
        "n_samples": total,
        "correct": n_correct,
        "accuracy": round(acc, 4),
        "by_relation_type": {
            rt: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"] if v["total"] else 0.0, 4),
            }
            for rt, v in sorted(by_rt.items())
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("metrics: accuracy=%.4f (%d/%d)", acc, n_correct, total)
    for rt, v in sorted(by_rt.items()):
        a = v["correct"] / v["total"] if v["total"] else 0.0
        log.info("  %-8s : %.4f (%d/%d)", rt, a, v["correct"], v["total"])
    return metrics


def run_one(
    predictions_file: Path,
    evidence_map: dict[str, str],
    rt_map: dict[str, str],
    method: str,
    prompt_key: str,
    prompt_template: str,
    judge_model: str,
    num_runs: int,
    concurrency: int,
    timeout: int,
    max_retries: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = json.load(predictions_file.open(encoding="utf-8"))
    log.info(
        "===== JUDGE method=%s prompt=%s model=%s =====  "
        "n=%d  num_runs=%d  concurrency=%d",
        method, prompt_key, judge_model, len(predictions), num_runs, concurrency,
    )

    # Idempotency: skip if an identical judge.json already exists.
    judge_path = out_dir / "judge.json"
    if judge_path.exists():
        try:
            old = json.load(judge_path.open(encoding="utf-8"))
            same = (
                old.get("num_runs") == num_runs
                and old.get("judge_model") == judge_model
                and old.get("prompt") == prompt_key
                and len(old.get("details", [])) == len(predictions)
            )
            if same:
                log.info("[skip] judge.json already matches; rewriting metrics only")
                return _write_metrics(out_dir, old, rt_map)
        except Exception:
            pass

    all_run_labels: list[list[dict[str, Any]]] = []
    for run_idx in range(1, num_runs + 1):
        log.info("run %d/%d ...", run_idx, num_runs)
        run_results: list[dict[str, Any] | None] = [None] * len(predictions)

        def _task(i_item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            i, item = i_item
            sid = item.get("sample_id", "")
            return i, judge_one(
                question=item.get("question_input", ""),
                evidence=evidence_map.get(sid, ""),
                pred=item.get("prediction", ""),
                prompt_template=prompt_template,
                model=judge_model,
                timeout=timeout,
                max_retries=max_retries,
            )

        lock = threading.Lock()
        completed = 0
        start = datetime.now()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_task, (i, it)): i for i, it in enumerate(predictions)}
            for fut in as_completed(futs):
                try:
                    i, r = fut.result()
                except Exception as e:  # pragma: no cover - defensive
                    i = futs[fut]
                    r = {"label": "wrong", "reason": f"worker crash: {e}"}
                run_results[i] = r
                with lock:
                    completed += 1
                    if completed % 50 == 0 or completed == len(predictions):
                        elapsed = (datetime.now() - start).total_seconds()
                        log.info(
                            "  [run %d] %d/%d  elapsed=%.0fs",
                            run_idx, completed, len(predictions), elapsed,
                        )
        all_run_labels.append([
            (r if r is not None else {"label": "wrong", "reason": "missing"})
            for r in run_results
        ])

    # Per-sample majority vote (num_runs=1 => single-run label).
    details = []
    for i, item in enumerate(predictions):
        sid = item.get("sample_id", "")
        labels = [run[i]["label"] for run in all_run_labels]
        n_correct = labels.count("correct")
        final = "correct" if n_correct > num_runs / 2 else "wrong"
        details.append({
            "index": i,
            "sample_id": sid,
            "relation_type": rt_map.get(sid, "unknown"),
            "question_input": item.get("question_input", ""),
            "evidence": evidence_map.get(sid, ""),
            "prediction": item.get("prediction", ""),
            "labels_per_run": labels,
            "correct_count": n_correct,
            "final_label": final,
            "reasons_per_run": [run[i].get("reason", "") for run in all_run_labels],
        })

    judged = {
        "method": method,
        "prompt": prompt_key,
        "judge_model": judge_model,
        "num_runs": num_runs,
        "n_samples": len(predictions),
        "details": details,
    }
    judge_path.write_text(
        json.dumps(judged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("-> %s", judge_path)
    return _write_metrics(out_dir, judged, rt_map)


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Locomo-Plus memory-awareness judge.")
    ap.add_argument(
        "--predictions", type=str, required=True,
        help=(
            "Path to the predictions.json file produced by stage8_qa_locomo_plus, "
            "e.g. <experiment_dir>/predictions.json. The judge reads this file "
            "directly; no mount step under benchmark_eval/locomo_plus/predictions/ "
            "is required anymore."
        ),
    )
    ap.add_argument(
        "--method-tag", type=str, required=True,
        help=(
            "Display tag used only for the output sub-directory name "
            "(<results-root>/<method-tag>__<prompt>__<model_slug>/). "
            "Has no effect on the predictions path; supply something like "
            "'t_mem__<exp_name>__top10_summary'."
        ),
    )
    ap.add_argument(
        "--prompt", type=str, required=True,
        choices=sorted(PROMPT_REGISTRY.keys()),
        help="Judge prompt key (A_paper_original).",
    )
    ap.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=(
            "LLM model name passed through to the judge LLM. Default comes from "
            "T_mem.config.MODELS['locomo_plus_judge'] (" + DEFAULT_JUDGE_MODEL + ")."
        ),
    )
    ap.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument(
        "--results-root", type=str, default=str(LOCOMO_PLUS_DIR / "results"),
        help="Root directory for outputs (default: <locomo_plus>/results).",
    )
    args = ap.parse_args()

    # Resolve paths.
    pred_file = Path(args.predictions).resolve()
    if not pred_file.exists():
        log.error("predictions file not found: %s", pred_file)
        sys.exit(1)

    # Build evidence + relation_type maps once.
    predictions = json.load(pred_file.open(encoding="utf-8"))
    evidence_map, rt_map = build_evidence_and_reltype_maps(predictions)
    log.info(
        "loaded predictions=%d  evidence=%d  rt=%d",
        len(predictions), len(evidence_map), len(rt_map),
    )

    # Output dir: results/<method-tag>__<prompt>__<model_slug>/
    out_dir = (
        Path(args.results_root)
        / f"{args.method_tag}__{args.prompt}__{_model_slug(args.judge_model)}"
    )

    prompt_template = get_prompt(args.prompt)
    run_one(
        predictions_file=pred_file,
        evidence_map=evidence_map,
        rt_map=rt_map,
        method=args.method_tag,
        prompt_key=args.prompt,
        prompt_template=prompt_template,
        judge_model=args.judge_model,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
