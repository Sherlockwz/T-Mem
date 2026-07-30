#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_hypermem.py — HyperMem-style QA stage for T-Mem LongMemEval evaluation.

差异对照原有 stage8_qa_lme.py：
  1. Answer prompt → HyperMem 7-step CoT（要求 FINAL ANSWER 段）
  2. 答案后处理 → 提取 "FINAL ANSWER:" 之后的文本（HyperMem 风格）
  3. 输出文件 → hypothesis_hypermem.jsonl + responses_hypermem.json（不覆盖原有文件）

其余逻辑（数据加载、断点续跑、并发控制）100% 复用原有 stage8 的链路。

Usage:
  python3 qa_hypermem.py --memory-dir <dir> --out-dir <dir> [--stitched-file <path>] [--concurrency N]
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

# ---- self-contained: ensure T_mem is importable ----
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent  # T_Mem_final_git_0602
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.llm.venus_provider import VenusLLMProvider  # noqa: E402
from T_mem.persona.qa_support import (  # noqa: E402
    append_persona_section,
    load_persona_block,
    parse_persona_store_root_env,
    parse_speakers_from_ctx,
)

# ---- local prompt (self-contained, no EverOS dependency) ----
from hypermem_prompts import ANSWER_PROMPT_HYPERMEM  # noqa: E402

_logger = logging.getLogger("hypermem.qa")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = int(os.environ.get("T_MEM_QA_CONCURRENCY", "14"))
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# helpers — same as stage8_qa_lme.py
# ---------------------------------------------------------------------------

def _conv_id_from_user_key(user_key: str) -> Optional[int]:
    try:
        return int(user_key.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _load_lme_meta(stitched_file: Path) -> Dict[int, Dict[str, str]]:
    data = json.loads(stitched_file.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, str]] = {}
    for rec in data:
        idx = rec.get("sample_idx")
        if idx is None:
            continue
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


# ---------------------------------------------------------------------------
# QA — HyperMem style: 7-step CoT + FINAL ANSWER extraction
# ---------------------------------------------------------------------------

async def _answer_one_hypermem(provider: VenusLLMProvider, prompt: str) -> str:
    """Ask LLM once with retry; extract text after 'FINAL ANSWER:' (HyperMem style).

    Returns '' on persistent failure.
    """
    last_err: str = ""
    for i in range(MAX_RETRIES):
        try:
            raw = await provider.generate(prompt, temperature=0.0)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            _logger.warning(
                "[hypermem-qa] attempt %d/%d error: %s", i + 1, MAX_RETRIES, last_err,
            )
            continue
        if not raw:
            continue

        # HyperMem post-processing: extract text after FINAL ANSWER:
        raw_stripped = raw.strip()
        if "FINAL ANSWER:" in raw_stripped:
            parts = raw_stripped.split("FINAL ANSWER:")
            if len(parts) > 1:
                return parts[1].strip()
        # Fallback: use full raw response (mirrors HyperMem fallback)
        return raw_stripped
    return ""


async def _answer_record_hypermem(
    provider: VenusLLMProvider,
    record: Dict[str, Any],
    *,
    user_key: str,
    conv_id: Optional[int],
    lme_meta: Dict[int, Dict[str, str]],
    persona_store_root: Optional[Path] = None,
    sem: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
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

    # Use HyperMem 7-step CoT prompt
    prompt = ANSWER_PROMPT_HYPERMEM.format(
        context=ctx,
        question=query,
    )

    async with sem:
        try:
            hyp = await _answer_one_hypermem(provider, prompt)
        except Exception as e:
            _logger.error(
                "[hypermem-qa][%s] answer failed for qid=%s: %s: %s",
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
        "answer":        meta["answer"],
        "hypothesis":    hyp,
        "model":         provider.model if hasattr(provider, "model") else "",
    }


def _write_responses_grouped(
    responses_path: Path,
    responses_out: Dict[str, List[Dict[str, Any]]],
) -> None:
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    memory_dir = Path(args.memory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support non-default search_results file (for ablation experiments).
    search_file = getattr(args, "search_file", None)
    if search_file:
        search_path = Path(search_file)
        if not search_path.is_absolute():
            search_path = memory_dir / search_path
        search_path = search_path.resolve()
    else:
        search_path = memory_dir / "search_results.json"
    if not search_path.exists():
        raise FileNotFoundError(f"search_results missing: {search_path}")

    stitched_default = memory_dir / "data" / "stitched_lme.json"
    stitched_path = Path(args.stitched_file).resolve() if args.stitched_file \
        else stitched_default
    if not stitched_path.exists():
        raise FileNotFoundError(
            f"stitched_lme.json missing: {stitched_path}\n"
            "Run T_mem.main.stage0_lme_stitch first."
        )

    persona_store_root = None
    if getattr(args, "no_persona", False):
        persona_store_root = None
    else:
        persona_store_root = parse_persona_store_root_env()

    _logger.info("[hypermem-qa] memory_dir    = %s", memory_dir)
    _logger.info("[hypermem-qa] out_dir       = %s", out_dir)
    _logger.info("[hypermem-qa] search_file   = %s", search_path)
    _logger.info("[hypermem-qa] stitched_file = %s", stitched_path)
    _logger.info("[hypermem-qa] model         = %s", args.model)
    _logger.info("[hypermem-qa] concurrency   = %d", args.concurrency)
    _logger.info("[hypermem-qa] prompt        = HyperMem 7-step CoT + FINAL ANSWER extraction")
    _logger.info(
        "[hypermem-qa] persona_store_root = %s",
        persona_store_root if persona_store_root is not None else "<disabled>",
    )

    lme_meta = _load_lme_meta(stitched_path)
    _logger.info("[hypermem-qa] loaded LME meta for %d convs", len(lme_meta))

    search_results: Dict[str, List[Dict[str, Any]]] = json.loads(
        search_path.read_text(encoding="utf-8")
    )
    total_records = sum(len(v or []) for v in search_results.values())
    _logger.info(
        "[hypermem-qa] loaded search_results: %d users, %d records",
        len(search_results), total_records,
    )

    responses_path = out_dir / "responses_hypermem.json"
    hyp_path = out_dir / "hypothesis_hypermem.jsonl"

    # Resume
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
                    "[hypermem-qa] resuming: %d answered entries carried over", n_done,
                )
        except Exception as e:
            _logger.warning("[hypermem-qa] resume failed (%s); starting fresh", e)
            responses_out = {}
            already = {}

    provider = VenusLLMProvider(model=args.model, temperature=0.0)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    done_counter = sum(len(s) for s in already.values())
    last_report = time.time()
    report_every = 20

    async def _process(user_key: str, record: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        nonlocal done_counter, last_report
        conv_id = _conv_id_from_user_key(user_key)
        resp = await _answer_record_hypermem(
            provider,
            record,
            user_key=user_key,
            conv_id=conv_id,
            lme_meta=lme_meta,
            persona_store_root=persona_store_root,
            sem=sem,
        )
        done_counter += 1
        now = time.time()
        if done_counter % report_every == 0 or now - last_report > 30:
            _logger.info(
                "[hypermem-qa] progress %d/%d (%.1f%%)",
                done_counter, total_records,
                100.0 * done_counter / max(1, total_records),
            )
            last_report = now
        return user_key, resp

    tasks: List[asyncio.Task] = []
    for user_key, records in search_results.items():
        if not isinstance(records, list):
            continue
        seen = already.get(user_key, set())
        responses_out.setdefault(user_key, [])
        conv_id = _conv_id_from_user_key(user_key)
        meta = lme_meta.get(conv_id) if conv_id is not None else None
        target_qid = (meta or {}).get("question_id", "")
        for record in records:
            if target_qid and target_qid in seen:
                continue
            tasks.append(asyncio.create_task(_process(user_key, record)))

    _logger.info("[hypermem-qa] scheduling %d tasks", len(tasks))

    if tasks:
        for coro in asyncio.as_completed(tasks):
            try:
                user_key, resp = await coro
            except Exception as e:
                _logger.error("[hypermem-qa] task crashed: %s: %s",
                              type(e).__name__, e)
                continue
            if resp is None:
                continue
            responses_out.setdefault(user_key, []).append(resp)
            if done_counter % 50 == 0:
                _write_responses_grouped(responses_path, responses_out)

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
        "[hypermem-qa] DONE: %d records (%d answered / %d empty / %d hypothesis lines) -> %s + %s",
        n_total, n_answered, n_total - n_answered, n_hyp,
        responses_path, hyp_path,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HyperMem-style LongMemEval QA — 7-step CoT + FINAL ANSWER extraction",
    )
    ap.add_argument("--memory-dir", required=True,
                    help="Experiment dir containing search_results.json")
    ap.add_argument("--out-dir", required=True,
                    help="Where responses_hypermem.json + hypothesis_hypermem.jsonl will be written")
    ap.add_argument("--stitched-file", default="",
                    help="Override path to stitched_lme.json (default: <memory_dir>/data/stitched_lme.json)")
    ap.add_argument("--search-file", default="",
                    help="Override search_results file path (default: <memory_dir>/search_results.json). "
                         "Use for ablation experiments with mutated context.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Reader model alias (default: gpt-4o-mini)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--no-persona", action="store_true",
                    help="Disable persona profile injection (for ablation experiments)")
    ap.add_argument("--force-restart", action="store_true",
                    help="Ignore existing responses and re-answer everything")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
