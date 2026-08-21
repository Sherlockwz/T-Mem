"""Stage 8 (LongMemEval): per-instance QA over stage6 retrieval output.

100 % aligned with the official LongMemEval reader
(`LongMemEval-main/src/generation/run_generation.py`):
  - Prompt: byte-for-byte the official `con` (CoT) reader template, exposed as
    `ANSWER_PROMPT_LME_COT` in `T_mem.prompts.answer_prompts`.
  - Model: gpt-4o-mini (LongMemEval official baseline reader).
  - Hypothesis is the **raw model response** (no NEMORI-style FINAL ANSWER
    parsing): the upstream baseline never strips the "Answer (step by step):"
    output, and the judge prompts are written to tolerate intermediate
    reasoning. Doing custom postprocessing here would diverge from the
    official protocol -- explicitly forbidden.

Inputs (consumed verbatim):
  <memory_dir>/search_results.json   -- stage6 main retrieval output, keyed
                                        by `f"locomo_exp_user_{conv_id}"`,
                                        each value a list of {query, context}
                                        records. For LongMemEval each conv
                                        has exactly one record (one question
                                        per stitched instance).
  <memory_dir>/data/stitched_lme.json
                                     -- output of stage0_lme_stitch; provides
                                        the conv_id -> lme_question_id /
                                        question_date / question_type
                                        mapping. Required because
                                        search_results.json carries only the
                                        question text, not the LME metadata.

Outputs:
  <out_dir>/responses.json    -- LoCoMo-shape grouped responses, one entry per
                                 (conv_id, qa) pair, containing the LongMemEval
                                 metadata for human / debug inspection. Schema
                                 mirrors stage8_qa_locomo's responses.json so
                                 existing tooling (resume, logs) keeps working.
  <out_dir>/hypothesis.jsonl  -- LongMemEval official hypothesis schema,
                                 one JSON object per line: `{question_id,
                                 hypothesis}`. This is the file fed into
                                 the official judge.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
_T_MEM_ROOT = _HERE.parent.parent.parent
if str(_T_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_T_MEM_ROOT))

from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.llm.llm_provider import LLMProvider  # noqa: E402
from T_mem.prompts.answer_prompts import (  # noqa: E402
    ANSWER_PROMPT_LME,
    ANSWER_PROMPT_LME_COT,
)
from T_mem.persona.qa_support import (  # noqa: E402
    append_persona_section,
    load_persona_block,
    parse_persona_store_root_env,
    parse_speakers_from_ctx,
)
from T_mem.config import MODELS  # noqa: E402

_logger = logging.getLogger("T_mem.evaluation.stage8_qa_lme")

# Reading-method registry mirrors the upstream `--cot true|false` switch
# (the `--con` separate-extract mode is intentionally NOT exposed because it
# requires a second LLM round per chunk, which we do not run here).
ANSWER_PROMPT_TEMPLATES: Dict[str, str] = {
    "lme":     ANSWER_PROMPT_LME,
    "lme_cot": ANSWER_PROMPT_LME_COT,
}
DEFAULT_ANSWER_PROMPT = "lme_cot"  # == upstream reading_method=con

# LongMemEval official baseline reader = gpt-4o-mini-2024-07-18; we route
# through `MODELS["locomo_qa"]` which is already gpt-4o-mini (config.py).
# Override via --model if a different alias is needed.
DEFAULT_MODEL = MODELS["locomo_qa"]
DEFAULT_CONCURRENCY = int(os.environ.get("T_MEM_QA_CONCURRENCY", "14"))
MAX_RETRIES = 5


def _conv_id_from_user_key(user_key: str) -> Optional[int]:
    """`locomo_exp_user_3` -> 3; None for malformed keys."""
    try:
        return int(user_key.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _load_lme_meta(stitched_file: Path) -> Dict[int, Dict[str, str]]:
    """conv_id -> {question_id, question_type, question_date, answer, question}.

    `conv_id` is the stitched record's `sample_idx` (0-indexed, matches
    stage1's `for con_id, conversation in enumerate(conversations)`).
    """
    data = json.loads(stitched_file.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, str]] = {}
    for rec in data:
        idx = rec.get("sample_idx")
        if idx is None:
            continue
        # Defensive: re-read question / answer from `qa[0]` so any future
        # divergence between top-level lme_* and qa[] is caught early.
        qa_list = rec.get("qa") or []
        question = (qa_list[0].get("question") if qa_list else "") or ""
        answer   = (qa_list[0].get("answer")   if qa_list else "") or ""
        out[int(idx)] = {
            "question_id":   rec.get("lme_question_id") or "",
            "question_type": rec.get("lme_question_type") or "",
            "question_date": rec.get("lme_question_date") or "",
            "answer":        rec.get("lme_answer") or answer,
            "question":      question,
        }
    return out

def _assemble_context(
    base_ctx: str,
    *,
    persona_store_root: Optional[Path],
    conv_id: Optional[int],
) -> str:
    """Append persona block to base_ctx when configured; no-op on missing/empty."""
    if persona_store_root is None or conv_id is None:
        return base_ctx
    hint_a, hint_b = parse_speakers_from_ctx(base_ctx)
    persona_md, _diag = load_persona_block(
        persona_store_root,
        conv_id=conv_id,
        speaker_hint_a=hint_a,
        speaker_hint_b=hint_b,
    )
    if not persona_md:
        return base_ctx
    return append_persona_section(base_ctx, persona_md)

async def _answer_one(provider: LLMProvider, prompt: str) -> str:
    """Ask the LLM once with retry; returns '' on persistent failure.

    NOTE: unlike stage8_qa_locomo which strips around `FINAL ANSWER:`, the
    LongMemEval reader's CoT output is FREE-FORM. The official baseline
    feeds the entire raw model response into the judge, so we do too --
    any custom postprocessing would diverge from the upstream protocol.
    """
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
        if not raw:
            continue
        return raw.strip()
    return ""


async def _answer_record(
    provider: LLMProvider,
    record: Dict[str, Any],
    *,
    user_key: str,
    conv_id: Optional[int],
    lme_meta: Dict[int, Dict[str, str]],
    answer_prompt_template: str,
    persona_store_root: Optional[Path] = None,
    sem: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Produce one response dict per (user_key, record). None when skipped."""
    query = (record.get("query") or "").strip()
    ctx = record.get("context") or ""

    # Append persona profile when configured.
    ctx = _assemble_context(
        ctx,
        persona_store_root=persona_store_root,
        conv_id=conv_id,
    )

    if not query or conv_id is None:
        return {
            "user_key":      user_key,
            "conv_id":       conv_id,
            "question_id":   "",
            "question_type": "",
            "question_date": "",
            "question":      query,
            "answer":        "",
            "hypothesis":    "",
            "model":         provider.model if hasattr(provider, "model") else "",
            "_error":        "empty_query_or_conv_id",
        }

    meta = lme_meta.get(conv_id)
    if meta is None:
        # search_results contains a conv that the stitched file does not
        # know about -- this is a build-data mismatch and we surface it
        # explicitly rather than silently fall back.
        return {
            "user_key":      user_key,
            "conv_id":       conv_id,
            "question_id":   "",
            "question_type": "",
            "question_date": "",
            "question":      query,
            "answer":        "",
            "hypothesis":    "",
            "model":         provider.model if hasattr(provider, "model") else "",
            "_error":        f"no_lme_meta_for_conv_{conv_id}",
        }

    prompt = answer_prompt_template.format(
        context=ctx,
        question_date=meta["question_date"],
        question=query,
    )

    async with sem:
        try:
            hyp = await _answer_one(provider, prompt)
        except Exception as e:  # noqa: BLE001
            _logger.error(
                "[%s] answer failed for qid=%s: %s: %s",
                user_key, meta["question_id"], type(e).__name__, e,
            )
            hyp = ""

    return {
        "user_key":      user_key,
        "conv_id":       conv_id,
        "question_id":   meta["question_id"],
        "question_type": meta["question_type"],
        "question_date": meta["question_date"],
        "question":      query,
        "answer":        meta["answer"],   # gold (kept for human inspection only)
        "hypothesis":    hyp,
        "model":         provider.model if hasattr(provider, "model") else "",
    }


def _write_responses_grouped(
    responses_path: Path,
    responses_out: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Atomic write of responses.json (LoCoMo-shape grouped)."""
    tmp = responses_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(responses_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(responses_path)


def _write_hypothesis_jsonl(
    hyp_path: Path,
    responses_out: Dict[str, List[Dict[str, Any]]],
) -> int:
    """Emit LongMemEval official hypothesis.jsonl (one {question_id, hypothesis} per line).

    Order: by `question_id` ascending so reruns are diff-stable. Records
    without a `question_id` are skipped (build-data mismatch).
    """
    rows: List[Tuple[str, str]] = []
    for items in responses_out.values():
        for r in items:
            qid = (r.get("question_id") or "").strip()
            hyp = r.get("hypothesis") or ""
            if not qid:
                continue
            rows.append((qid, hyp))
    rows.sort(key=lambda x: x[0])

    tmp = hyp_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid, hyp in rows:
            f.write(json.dumps(
                {"question_id": qid, "hypothesis": hyp},
                ensure_ascii=False,
            ) + "\n")
    tmp.replace(hyp_path)
    return len(rows)


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    memory_dir = Path(args.memory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    search_path = memory_dir / "search_results.json"
    if not search_path.exists():
        raise FileNotFoundError(f"search_results missing: {search_path}")

    # Stitched file is the source of truth for LongMemEval metadata.
    stitched_default = memory_dir / "data" / "stitched_lme.json"
    stitched_path = Path(args.stitched_file).resolve() if args.stitched_file \
        else stitched_default
    if not stitched_path.exists():
        raise FileNotFoundError(
            f"stitched_lme.json missing: {stitched_path}\n"
            "Run T_mem.main.stage0_lme_stitch first."
        )

    answer_prompt_kind = args.answer_prompt
    if answer_prompt_kind not in ANSWER_PROMPT_TEMPLATES:
        raise ValueError(
            f"--answer-prompt must be one of {list(ANSWER_PROMPT_TEMPLATES)}, "
            f"got {answer_prompt_kind!r}"
        )
    answer_prompt_template = ANSWER_PROMPT_TEMPLATES[answer_prompt_kind]

    persona_store_root = None
    if getattr(args, "no_persona", False):
        persona_store_root = None
    else:
        persona_store_root = parse_persona_store_root_env()

    _logger.info("[stage8_lme] memory_dir         = %s", memory_dir)
    _logger.info("[stage8_lme] out_dir            = %s", out_dir)
    _logger.info("[stage8_lme] stitched_file      = %s", stitched_path)
    _logger.info("[stage8_lme] model              = %s", args.model)
    _logger.info("[stage8_lme] answer_prompt      = %s (== upstream %s)",
                 answer_prompt_kind,
                 "reading_method=con" if answer_prompt_kind == "lme_cot"
                 else "reading_method=direct")
    _logger.info("[stage8_lme] concurrency        = %d", args.concurrency)
    _logger.info(
        "[stage8_lme] persona_store_root = %s",
        persona_store_root if persona_store_root is not None else "<disabled>",
    )

    lme_meta = _load_lme_meta(stitched_path)
    _logger.info("[stage8_lme] loaded LME meta for %d convs", len(lme_meta))

    search_results: Dict[str, List[Dict[str, Any]]] = json.loads(
        search_path.read_text(encoding="utf-8")
    )
    total_records = sum(len(v or []) for v in search_results.values())
    _logger.info(
        "[stage8_lme] loaded search_results: %d users, %d records",
        len(search_results), total_records,
    )

    responses_path = out_dir / "responses.json"
    hyp_path = out_dir / "hypothesis.jsonl"

    # Resume: reuse already-answered (user, question_id) pairs.
    responses_out: Dict[str, List[Dict[str, Any]]] = {}
    already: Dict[str, set] = {}
    if responses_path.exists() and not args.force_restart:
        try:
            prev = json.loads(responses_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                for u, items in prev.items():
                    if not isinstance(items, list):
                        continue
                    responses_out[u] = list(items)
                    already[u] = {
                        (it.get("question_id") or "")
                        for it in items
                        if isinstance(it, dict) and (it.get("hypothesis") or "")
                    }
                n_done = sum(len(s) for s in already.values())
                _logger.info(
                    "[stage8_lme] resuming: %d answered entries carried over", n_done,
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "[stage8_lme] resume failed (%s); starting fresh", e,
            )
            responses_out = {}
            already = {}

    provider = LLMProvider(model=args.model, temperature=0.0)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    done_counter = sum(len(s) for s in already.values())
    last_report = time.time()
    report_every = 20

    async def _process(user_key: str, record: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        nonlocal done_counter, last_report
        conv_id = _conv_id_from_user_key(user_key)
        resp = await _answer_record(
            provider,
            record,
            user_key=user_key,
            conv_id=conv_id,
            lme_meta=lme_meta,
            answer_prompt_template=answer_prompt_template,
            persona_store_root=persona_store_root,
            sem=sem,
        )
        done_counter += 1
        now = time.time()
        if done_counter % report_every == 0 or now - last_report > 30:
            _logger.info(
                "[stage8_lme] progress %d/%d (%.1f%%)",
                done_counter, total_records,
                100.0 * done_counter / max(1, total_records),
            )
            last_report = now
        return user_key, resp

    tasks: List[asyncio.Task] = []
    for user_key, records in search_results.items():
        if not isinstance(records, list):
            continue
        # Resume key: question_id (more reliable than question text because
        # an instance may share question text across runs but question_id
        # is unique).
        seen = already.get(user_key, set())
        responses_out.setdefault(user_key, [])
        conv_id = _conv_id_from_user_key(user_key)
        meta = lme_meta.get(conv_id) if conv_id is not None else None
        target_qid = (meta or {}).get("question_id", "")
        for record in records:
            if target_qid and target_qid in seen:
                continue
            tasks.append(asyncio.create_task(_process(user_key, record)))

    _logger.info("[stage8_lme] scheduling %d tasks", len(tasks))

    if tasks:
        for coro in asyncio.as_completed(tasks):
            try:
                user_key, resp = await coro
            except Exception as e:  # noqa: BLE001
                _logger.error("[stage8_lme] task crashed: %s: %s",
                              type(e).__name__, e)
                continue
            if resp is None:
                continue
            responses_out.setdefault(user_key, []).append(resp)
            # Mid-run snapshot every 50 completed tasks so we can resume on kill.
            if done_counter % 50 == 0:
                _write_responses_grouped(responses_path, responses_out)

    # Sort per-user items by question_id so reruns produce stable output.
    for user_key, items in responses_out.items():
        items.sort(key=lambda r: (r.get("question_id") or ""))

    _write_responses_grouped(responses_path, responses_out)
    n_hyp = _write_hypothesis_jsonl(hyp_path, responses_out)

    n_total = sum(len(v) for v in responses_out.values())
    n_answered = sum(
        1 for items in responses_out.values()
        for r in items if (r.get("hypothesis") or "")
    )
    _logger.info(
        "[stage8_lme] DONE: %d records (%d answered / %d empty / %d hypothesis lines) "
        "-> %s + %s",
        n_total, n_answered, n_total - n_answered, n_hyp,
        responses_path, hyp_path,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "T_mem stage 8 (LongMemEval) -- per-instance QA over stage6 "
            "retrieval context, official LongMemEval reader prompt + hypothesis schema."
        ),
    )
    ap.add_argument("--memory-dir", required=True,
                    help="Experiment dir containing search_results.json (and data/stitched_lme.json by default)")
    ap.add_argument("--out-dir", required=True,
                    help="Where responses.json + hypothesis.jsonl will be written")
    ap.add_argument("--stitched-file", default="",
                    help=(
                        "Override path to stitched_lme.json. "
                        "Defaults to <memory_dir>/data/stitched_lme.json."
                    ))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Reader model alias passed to LLMProvider. "
                         "Default = T_mem.config.MODELS['locomo_qa'] (gpt-4o-mini)")
    ap.add_argument(
        "--answer-prompt",
        default=DEFAULT_ANSWER_PROMPT,
        choices=sorted(ANSWER_PROMPT_TEMPLATES.keys()),
        help="lme_cot == upstream reading_method=con (default); "
             "lme == upstream reading_method=direct.",
    )
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--no-persona", action="store_true",
                    help="Disable persona profile injection (for ablation experiments)")
    ap.add_argument("--force-restart", action="store_true",
                    help="Ignore existing responses.json and re-answer everything")
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
