"""Stage 8: end-to-end QA over stage6 retrieval output.
Reads memory_dir/search_results.json (+ optional persona via T_MEM_PERSONA_STORE_ROOT),
stitches retrieval ctx into answer prompt, writes out_dir/responses.json."""

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
    ANSWER_PROMPT_NEMORI,
    ANSWER_PROMPT_NEMORI_COT,
)
from T_mem.persona.qa_support import (  # noqa: E402
    append_persona_section,
    load_persona_block,
    parse_persona_store_root_env,
    parse_speakers_from_ctx,
)
from T_mem.config import MODELS  # noqa: E402

_logger = logging.getLogger("T_mem.evaluation.stage8_qa_locomo")

ANSWER_PROMPT_TEMPLATES: Dict[str, str] = {
    "nemori": ANSWER_PROMPT_NEMORI,
    "nemori_cot": ANSWER_PROMPT_NEMORI_COT,
}
DEFAULT_ANSWER_PROMPT = "nemori"

# CRITICAL: QA model id pulled from T_mem.config.MODELS — do NOT read from env.
DEFAULT_MODEL = MODELS["locomo_qa"]
DEFAULT_CONCURRENCY = int(os.environ.get("T_MEM_QA_CONCURRENCY", "14"))
MAX_RETRIES = 5


async def _answer_one(
    provider: LLMProvider,
    prompt: str,
    *,
    answer_prompt_kind: str,
) -> str:
    """Ask the LLM once with retry; returns '' on persistent failure.
    nemori_cot strips the preamble before 'FINAL ANSWER:'; nemori strips trailing 'Answer:'."""
    for i in range(MAX_RETRIES):
        try:
            raw = await provider.generate(prompt, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            _logger.warning("[gen] attempt %d/%d error: %s", i + 1, MAX_RETRIES, e)
            continue
        if not raw:
            continue
        if answer_prompt_kind == "nemori_cot":
            parts = raw.split("FINAL ANSWER:")
            if len(parts) > 1:
                result = parts[1].strip()
                if result:
                    return result
            continue
        result = raw.strip()
        if "Answer:" in result:
            result = result.rsplit("Answer:", 1)[-1].strip()
        if result:
            return result
    return ""


def _conv_id_from_user_key(user_key: str) -> Optional[int]:
    """locomo_exp_user_3 -> 3; None for malformed keys."""
    try:
        return int(user_key.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


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


async def _answer_record(
    provider: LLMProvider,
    record: Dict[str, Any],
    *,
    user_key: str,
    conv_id: Optional[int],
    persona_store_root: Optional[Path],
    answer_prompt_template: str,
    answer_prompt_kind: str,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Produce one response dict per record; answer='' on failure (caller needs every row)."""
    query = (record.get("query") or "").strip()
    ctx_raw = record.get("context") or ""

    if not query:
        return {
            "question": query,
            "answer": "",
            "model": provider.model if hasattr(provider, "model") else "",
            "_error": "empty_query",
        }

    ctx = _assemble_context(
        ctx_raw,
        persona_store_root=persona_store_root,
        conv_id=conv_id,
    )
    prompt = answer_prompt_template.format(context=ctx, question=query)

    async with sem:
        try:
            answer = await _answer_one(
                provider, prompt, answer_prompt_kind=answer_prompt_kind
            )
        except Exception as e:  # noqa: BLE001
            _logger.error(
                "[%s] answer failed for %r: %s: %s",
                user_key, query[:60], type(e).__name__, e,
            )
            answer = ""

    out = {
        "question": query,
        "answer": answer,
        "model": provider.model if hasattr(provider, "model") else "",
    }
    # [BENCHMARK ADD-ON] carry retrieval traceability from stage6 record so the
    # final responses.json ties each answer to the turns T-mem retrieved.
    for k in ("retrieved_turn_ids", "retrieved_scene_ids", "retrieved_item_ids",
              "turn_ids_from_scenes", "turn_ids_from_items"):
        if k in record:
            out[k] = record[k]
    return out


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    memory_dir = Path(args.memory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    search_path = memory_dir / "search_results.json"
    responses_path = out_dir / "responses.json"

    if not search_path.exists():
        raise FileNotFoundError(f"search_results missing: {search_path}")

    answer_prompt_kind = args.answer_prompt
    if answer_prompt_kind not in ANSWER_PROMPT_TEMPLATES:
        raise ValueError(
            f"--answer-prompt must be one of {list(ANSWER_PROMPT_TEMPLATES)}, "
            f"got {answer_prompt_kind!r}"
        )
    answer_prompt_template = ANSWER_PROMPT_TEMPLATES[answer_prompt_kind]

    persona_store_root = parse_persona_store_root_env()

    _logger.info("[stage8] memory_dir         = %s", memory_dir)
    _logger.info("[stage8] out_dir            = %s", out_dir)
    _logger.info("[stage8] model              = %s", args.model)
    _logger.info("[stage8] answer_prompt      = %s", answer_prompt_kind)
    _logger.info("[stage8] concurrency        = %d", args.concurrency)
    _logger.info(
        "[stage8] persona_store_root = %s",
        persona_store_root if persona_store_root is not None else "<disabled>",
    )

    search_results: Dict[str, List[Dict[str, Any]]] = json.loads(
        search_path.read_text(encoding="utf-8")
    )
    total_records = sum(len(v or []) for v in search_results.values())
    _logger.info(
        "[stage8] loaded search_results: %d users, %d records",
        len(search_results), total_records,
    )

    # Resume support: reuse already-answered (user, question) pairs; --force-restart wipes.
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
                        (it.get("question") or "").strip()
                        for it in items
                        if isinstance(it, dict) and (it.get("answer") or "")
                    }
                n_done = sum(len(s) for s in already.values())
                _logger.info(
                    "[stage8] resuming: %d answered entries carried over", n_done,
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "[stage8] resume failed (%s); starting fresh", e,
            )
            responses_out = {}
            already = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    provider = LLMProvider(model=args.model, temperature=0.0)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    done_counter = sum(len(s) for s in already.values())
    last_report = time.time()
    report_every = 20

    async def _process(user_key: str, idx: int, record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        nonlocal done_counter, last_report
        conv_id = _conv_id_from_user_key(user_key)
        resp = await _answer_record(
            provider,
            record,
            user_key=user_key,
            conv_id=conv_id,
            persona_store_root=persona_store_root,
            answer_prompt_template=answer_prompt_template,
            answer_prompt_kind=answer_prompt_kind,
            sem=sem,
        )
        done_counter += 1
        now = time.time()
        if done_counter % report_every == 0 or now - last_report > 30:
            _logger.info(
                "[stage8] progress %d/%d (%.1f%%)",
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
        for idx, record in enumerate(records):
            q = (record.get("query") or "").strip()
            if q and q in seen:
                continue  # already answered in a previous run
            tasks.append(asyncio.create_task(_process(user_key, idx, record)))

    _logger.info("[stage8] scheduling %d tasks", len(tasks))

    if tasks:
        for coro in asyncio.as_completed(tasks):
            try:
                user_key, resp = await coro
            except Exception as e:  # noqa: BLE001
                _logger.error("[stage8] task crashed: %s: %s", type(e).__name__, e)
                continue
            responses_out.setdefault(user_key, []).append(resp)

    # Sort per-user items by (question) so reruns produce stable output.
    for user_key, items in responses_out.items():
        items.sort(key=lambda r: (r.get("question") or ""))

    tmp_path = responses_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(responses_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(responses_path)

    n_total = sum(len(v) for v in responses_out.values())
    n_answered = sum(
        1 for items in responses_out.values()
        for r in items if (r.get("answer") or "")
    )
    _logger.info(
        "[stage8] DONE: wrote %d records (%d answered / %d empty) -> %s",
        n_total, n_answered, n_total - n_answered, responses_path,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "T_mem stage 8 — end-to-end QA over stage6 retrieval context."
        ),
    )
    ap.add_argument("--memory-dir", required=True,
                    help="Experiment dir containing search_results.json")
    ap.add_argument("--out-dir", required=True,
                    help="Where responses.json will be written")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--answer-prompt",
        default=DEFAULT_ANSWER_PROMPT,
        choices=sorted(ANSWER_PROMPT_TEMPLATES.keys()),
    )
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--force-restart", action="store_true",
                    help="Ignore existing responses.json and re-answer everything")
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    ap.add_argument("--tag", default="",
                    help="Free-form tag used only in log messages")
    args = ap.parse_args()

    if args.tag:
        _logger.info("[stage8] tag=%s", args.tag)

    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
