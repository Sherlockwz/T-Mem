"""Stage 8 (LoCoMo-Plus): per-sample QA over stage5_retrieval_locomo_plus top-K scenes.
Reads memory_dir/locomo_plus_topk_per_sample.json + locomo_plus.json + locomo10.json,
writes out_dir/predictions.json (one record per sample, run_judge.py-aligned schema)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve()
_T_MEM_ROOT = _HERE.parent.parent.parent
if str(_T_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_T_MEM_ROOT))

from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.llm.llm_provider import LLMProvider  # noqa: E402
from T_mem.config import MODELS  # noqa: E402

_logger = logging.getLogger("T_mem.evaluation.stage8_qa_locomo_plus")

# Canonical LoCoMo-Plus answer prompt. The judge in
# benchmark_eval/locomo_plus/judge/ scores responses produced by this exact
# wording, so keep the whitespace and the {speaker_a}/{speaker_b}/
# {memory_context}/{trigger} placeholders unchanged. Do NOT reformat.
QA_COGNITIVE_WITH_CUE_PROMPT = """You are continuing a conversation between {speaker_a} and {speaker_b}.

{memory_context}

Latest message from {speaker_a}: {trigger}

You must respond as {speaker_b}, drawing on the recalled experience above. Do NOT ignore or contradict the recalled information. Weave it naturally into your response.

Respond as {speaker_b}:"""

CONTENTS = ("summary", "description", "dialogue")

_AB_STRIP_PATTERN = re.compile(r'^\s*[AB]:\s*')


def _strip_ab_prefix(text: str) -> str:
    if not text:
        return text
    return _AB_STRIP_PATTERN.sub('', text, count=1)


def _format_sample_id(idx: int) -> str:
    return f"sample_{idx:03d}"


DEFAULT_MODEL = MODELS["locomo_plus_qa"]
DEFAULT_CONCURRENCY = int(os.environ.get("T_MEM_QA_CONCURRENCY", "12"))
MAX_RETRIES = 4


def _build_memory_context(topk_rows: List[Dict[str, Any]], content: str) -> str:
    """Stitch top-K rows into a '## Recalled Experience' block; content ∈ {summary, description, dialogue}."""
    blocks = ["## Recalled Experience"]
    for r in topk_rows:
        rank = r["rank"]
        if content == "summary":
            body = r.get("summary", "")
        elif content == "description":
            body = r.get("scene_description", "")
        elif content == "dialogue":
            body = r.get("original_dialogue", "")
        else:
            raise ValueError(f"unknown content: {content}")
        body = (body or "").strip() or "(empty)"
        blocks.append(f"[Memory {rank}]\n{body}")
    return "\n\n".join(blocks)


def _load_speaker_map_from_stitched(
    stitched_file: Path, n_samples: int,
) -> Dict[int, Tuple[str, str]]:
    """sample_idx -> (speaker_a, speaker_b) via stitched_locomo_plus.json.

    Per-sample mode: the stitched dataset *is* the source of truth -- conv i
    is the dedicated conversation for plus_sample i, and its speaker_a /
    speaker_b were copied from base_conv = i % 10 at stitch time.
    """
    data = json.load(stitched_file.open("r", encoding="utf-8"))
    out: Dict[int, Tuple[str, str]] = {}
    n = min(len(data), n_samples)
    for sidx in range(n):
        conv = data[sidx].get("conversation", {}) or {}
        out[sidx] = (
            conv.get("speaker_a", "A") or "A",
            conv.get("speaker_b", "B") or "B",
        )
    return out


def _load_speaker_map_from_locomo10(
    locomo10_file: Path, n_samples: int,
) -> Dict[int, Tuple[str, str]]:
    """Legacy fallback: sample_idx -> (speaker_a, speaker_b) via sample_idx % n_conv.

    Kept only as a backward-compat path; per-sample mode should always
    prefer `_load_speaker_map_from_stitched`.
    """
    data = json.load(locomo10_file.open("r", encoding="utf-8"))
    n_conv = len(data)
    out: Dict[int, Tuple[str, str]] = {}
    for sidx in range(n_samples):
        conv = data[sidx % n_conv].get("conversation", {}) or {}
        out[sidx] = (
            conv.get("speaker_a", "A") or "A",
            conv.get("speaker_b", "B") or "B",
        )
    return out


def _load_trigger_map(locomo_plus_file: Path) -> Dict[int, Tuple[str, str]]:
    """sample_idx -> (relation_type, trigger_query_stripped)."""
    samples = json.load(locomo_plus_file.open("r", encoding="utf-8"))
    out: Dict[int, Tuple[str, str]] = {}
    for sidx, s in enumerate(samples):
        tq = _strip_ab_prefix((s.get("trigger_query") or "").strip())
        rt = s.get("relation_type") or "unknown"
        out[sidx] = (rt, tq)
    return out


async def _answer_one(
    provider: LLMProvider,
    prompt: str,
) -> str:
    """Ask the LLM once with retry; returns an error marker on persistent failure."""
    last_err: str = ""
    for i in range(MAX_RETRIES):
        try:
            raw = await provider.generate(prompt, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            _logger.warning(
                "[gen] attempt %d/%d error: %s", i + 1, MAX_RETRIES, last_err,
            )
            continue
        if raw:
            return raw.strip()
    return f"(Error: {last_err})" if last_err else ""


async def _answer_record(
    provider: LLMProvider,
    *,
    sample_id: str,
    trigger: str,
    speaker_a: str,
    speaker_b: str,
    topk_rows: List[Dict[str, Any]],
    content: str,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    memory_context = _build_memory_context(topk_rows, content)
    prompt = QA_COGNITIVE_WITH_CUE_PROMPT.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        memory_context=memory_context,
        trigger=trigger,
    )
    t0 = time.perf_counter()
    async with sem:
        try:
            pred = await _answer_one(provider, prompt)
        except Exception as e:  # noqa: BLE001
            pred = f"(Worker crash: {type(e).__name__}: {e})"
    latency = round(time.perf_counter() - t0, 3)
    return {
        "sample_id": sample_id,
        "question_input": trigger,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "prediction": pred,
        "model": provider.model if hasattr(provider, "model") else "",
        "latency_s": latency,
        "prompt_len": len(prompt),
        "memory_context_len": len(memory_context),
    }


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    memory_dir = Path(args.memory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    topk_file = memory_dir / "locomo_plus_topk_per_sample.json"
    if not topk_file.exists():
        raise FileNotFoundError(
            f"locomo_plus_topk_per_sample.json missing: {topk_file}\n"
            "Run T_mem.main.stage5_retrieval_locomo_plus first."
        )

    project_root = _T_MEM_ROOT
    locomo_plus_file = Path(
        args.locomo_plus_file or os.environ.get(
            "T_MEM_LOCOMO_PLUS_FILE",
            str(project_root / "benchmark_eval" / "locomo_plus" / "data" / "locomo_plus.json"),
        )
    )
    if not locomo_plus_file.exists():
        raise FileNotFoundError(f"locomo_plus file missing: {locomo_plus_file}")

    # Speaker source priority:
    #   1) --stitched-file / T_MEM_STITCHED_PLUS_FILE  (per-sample mode)
    #   2) --locomo10-file / T_MEM_LOCOMO10_FILE       (legacy idx%n_conv)
    stitched_file_str = args.stitched_file or os.environ.get(
        "T_MEM_STITCHED_PLUS_FILE", "",
    )
    locomo10_file_str = args.locomo10_file or os.environ.get(
        "T_MEM_LOCOMO10_FILE", "",
    )
    stitched_file = Path(stitched_file_str).resolve() if stitched_file_str else None
    locomo10_file = Path(locomo10_file_str).resolve() if locomo10_file_str else None
    if stitched_file is None and locomo10_file is None:
        # last-resort default: locomo10.json shipped with the repo
        locomo10_file = Path(
            project_root / "benchmark_eval" / "locomo" / "data" / "locomo10.json"
        ).resolve()

    if args.content not in CONTENTS:
        raise ValueError(
            f"--content must be one of {CONTENTS}, got {args.content!r}"
        )

    _logger.info("[stage8_plus] memory_dir       = %s", memory_dir)
    _logger.info("[stage8_plus] out_dir          = %s", out_dir)
    _logger.info("[stage8_plus] topk_file        = %s", topk_file)
    _logger.info("[stage8_plus] locomo_plus_file = %s", locomo_plus_file)
    _logger.info("[stage8_plus] stitched_file    = %s", stitched_file)
    _logger.info("[stage8_plus] locomo10_file    = %s", locomo10_file)
    _logger.info("[stage8_plus] topk             = %d", args.topk)
    _logger.info("[stage8_plus] content          = %s", args.content)
    _logger.info("[stage8_plus] model            = %s", args.model)
    _logger.info("[stage8_plus] concurrency      = %d", args.concurrency)

    # Load inputs
    topk_payload = json.loads(topk_file.read_text(encoding="utf-8"))
    samples_map = topk_payload.get("samples", {}) or {}
    n_samples = len(samples_map)
    _logger.info("[stage8_plus] loaded %d samples from topk cache", n_samples)

    lp_samples = json.load(locomo_plus_file.open("r", encoding="utf-8"))
    n_lp = len(lp_samples)

    if stitched_file is not None and stitched_file.exists():
        speakers = _load_speaker_map_from_stitched(stitched_file, n_samples=n_lp)
        _logger.info(
            "[stage8_plus] speaker source = stitched (%s)", stitched_file,
        )
    else:
        if locomo10_file is None or not locomo10_file.exists():
            raise FileNotFoundError(
                "speaker source unresolved: neither --stitched-file nor "
                "--locomo10-file was provided / found."
            )
        speakers = _load_speaker_map_from_locomo10(locomo10_file, n_samples=n_lp)
        _logger.info(
            "[stage8_plus] speaker source = locomo10 [legacy idx %% n_conv] (%s)",
            locomo10_file,
        )
    triggers = _load_trigger_map(locomo_plus_file)

    # Resume support: skip samples that already have a non-error prediction.
    predictions_path = out_dir / "predictions.json"
    done: Dict[str, Dict[str, Any]] = {}
    if predictions_path.exists() and not args.force_restart:
        try:
            prev = json.loads(predictions_path.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                for rec in prev:
                    sid = rec.get("sample_id")
                    pred = rec.get("prediction") or ""
                    if sid and pred and "(Error" not in pred and "(Worker crash" not in pred:
                        done[sid] = rec
            if done:
                _logger.info(
                    "[stage8_plus] resuming: %d answered entries carried over",
                    len(done),
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "[stage8_plus] resume failed (%s); starting fresh", e,
            )
            done = {}

    provider = LLMProvider(model=args.model, temperature=0.0)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    # Schedule tasks for every sample that has both a topk entry and a valid
    # trigger_query. Missing trigger → skip with an explicit empty-pred marker.
    todo: List[str] = []
    tasks: List[asyncio.Task] = []
    for sidx in range(n_lp):
        sid = _format_sample_id(sidx)
        if sid in done:
            continue
        rows = samples_map.get(sid)
        if not rows:
            _logger.warning("[stage8_plus] no topk rows for %s; skipping", sid)
            done[sid] = {
                "sample_id": sid,
                "question_input": triggers.get(sidx, ("", ""))[1],
                "speaker_a": speakers.get(sidx, ("A", "B"))[0],
                "speaker_b": speakers.get(sidx, ("A", "B"))[1],
                "prediction": "(Error: no topk rows)",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_len": 0,
                "memory_context_len": 0,
            }
            continue
        _rt, tq = triggers.get(sidx, ("", ""))
        if not tq:
            _logger.warning("[stage8_plus] empty trigger_query for %s; skipping", sid)
            done[sid] = {
                "sample_id": sid,
                "question_input": "",
                "speaker_a": speakers.get(sidx, ("A", "B"))[0],
                "speaker_b": speakers.get(sidx, ("A", "B"))[1],
                "prediction": "(Error: empty trigger_query)",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_len": 0,
                "memory_context_len": 0,
            }
            continue

        sa, sb = speakers.get(sidx, ("A", "B"))
        rows_cut = rows[: args.topk]
        todo.append(sid)
        tasks.append(
            asyncio.create_task(
                _answer_record(
                    provider,
                    sample_id=sid,
                    trigger=tq,
                    speaker_a=sa,
                    speaker_b=sb,
                    topk_rows=rows_cut,
                    content=args.content,
                    sem=sem,
                )
            )
        )

    _logger.info(
        "[stage8_plus] scheduling %d tasks (resumed=%d)", len(tasks), len(done),
    )

    if tasks:
        start = time.time()
        completed = 0
        report_every = 20
        for coro in asyncio.as_completed(tasks):
            try:
                rec = await coro
            except Exception as e:  # noqa: BLE001
                _logger.error(
                    "[stage8_plus] task crashed: %s: %s", type(e).__name__, e,
                )
                continue
            done[rec["sample_id"]] = rec
            completed += 1
            if completed % report_every == 0 or completed == len(tasks):
                elapsed = time.time() - start
                _logger.info(
                    "[stage8_plus] progress %d/%d  elapsed=%.0fs",
                    completed, len(tasks), elapsed,
                )
                # Mid-run snapshot so we can resume on kill.
                _write_predictions(predictions_path, n_lp, done)

    _write_predictions(predictions_path, n_lp, done)

    n_err = sum(
        1 for r in done.values()
        if "(Error" in r.get("prediction", "")
        or "(Worker crash" in r.get("prediction", "")
    )
    _logger.info(
        "[stage8_plus] DONE: wrote %d records (%d errors) -> %s",
        len(done), n_err, predictions_path,
    )


def _write_predictions(
    path: Path,
    n_lp: int,
    done: Dict[str, Dict[str, Any]],
) -> None:
    """Write predictions.json in sample_idx order, one row per answered sample."""
    rows: List[Dict[str, Any]] = []
    for sidx in range(n_lp):
        sid = _format_sample_id(sidx)
        if sid in done:
            rows.append(done[sid])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "T_mem stage 8 (LoCoMo-Plus) — per-sample QA over stage5_retrieval_locomo_plus "
        "top-K scenes."
        ),
    )
    ap.add_argument("--memory-dir", required=True,
                    help="Experiment dir containing locomo_plus_topk_per_sample.json")
    ap.add_argument("--out-dir", required=True,
                    help="Where predictions.json will be written")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--topk", type=int, default=10,
                    help="Number of top-K rows to stitch into memory_context")
    ap.add_argument(
        "--content", default="summary", choices=list(CONTENTS),
        help="Scene content type stitched into memory_context",
    )
    ap.add_argument("--locomo-plus-file", default="",
                    help="Override path to locomo_plus.json")
    ap.add_argument("--stitched-file", default="",
                    help=(
                        "Path to stitched_locomo_plus.json (per-sample mode). "
                        "If provided, speaker_a/b are read from "
                        "stitched[i]['conversation']; this is the source of "
                        "truth in per-sample memory-library mode."
                    ))
    ap.add_argument("--locomo10-file", default="",
                    help=(
                        "Legacy speaker source: pulls speaker_a/b via "
                        "sample_idx %% len(locomo10). Used only if "
                        "--stitched-file is not provided."
                    ))
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--force-restart", action="store_true",
                    help="Ignore existing predictions.json and re-answer everything")
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
