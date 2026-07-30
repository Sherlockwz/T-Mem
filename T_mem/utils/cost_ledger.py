"""Fine-grained per-LLM-call cost ledger for T-Mem construction stages.

Design goals
------------
* ZERO overhead + ZERO behaviour change when disabled. The ledger is a no-op
  unless env ``T_MEM_COST_LOG`` points at a writable JSONL path.
* One append-only JSONL row per *successful model completion* (the finest
  granularity possible). Anything coarser -- per stage, per sub-stage, per
  conversation -- can be re-aggregated offline from these rows, but the
  reverse is impossible, so we log at the call level.
* Robust: any exception inside the ledger is swallowed so cost logging can
  never break the pipeline.

Row schema (one JSON object per line)
-------------------------------------
ts, pid, stage, call_site, conv_id, model,
prompt_tokens, completion_tokens, total_tokens,          # chosen (api if available else tiktoken)
prompt_tokens_tok, completion_tokens_tok,                # tiktoken (always)
prompt_tokens_api, completion_tokens_api, total_tokens_api,  # api usage if returned, else null
token_source ("api"|"tiktoken"), enc, prompt_chars, completion_chars, latency_s

Threading of stage / call_site / conv_id
-----------------------------------------
* ``stage``    : env ``T_MEM_COST_STAGE`` (set once per stage *process* by the
                 run script). Coarse but 100% reliable.
* ``call_site``: passed explicitly by each ``.generate(..., call_site=...)``
                 site (fine sub-stage), or via the ``_substage_ctx`` contextvar.
* ``conv_id``  : set at each per-conversation entry point via ``set_conv()``.
                 asyncio stages read it in the async task (in generate()) and
                 pass it down through ``meta``; the synchronous persona stage
                 reads it directly (same thread).
"""
from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

# --- contextvars for conv / sub-stage attribution (async-safe) ---------------
_conv_ctx: ContextVar[Optional[str]] = ContextVar("tmem_cost_conv", default=None)
_substage_ctx: ContextVar[Optional[str]] = ContextVar("tmem_cost_substage", default=None)

_LOCK = threading.Lock()
_ENC = None
_ENC_NAME: Optional[str] = None
_ENC_TRIED = False


def _log_path() -> str:
    return os.environ.get("T_MEM_COST_LOG", "").strip()


def enabled() -> bool:
    return bool(_log_path())


def stage_label() -> str:
    return os.environ.get("T_MEM_COST_STAGE", "").strip() or "unknown"


def set_conv(conv_id) -> None:
    try:
        _conv_ctx.set(str(conv_id) if conv_id is not None else None)
    except Exception:
        pass


def get_conv() -> Optional[str]:
    try:
        return _conv_ctx.get()
    except Exception:
        return None


def set_substage(name: Optional[str]) -> None:
    try:
        _substage_ctx.set(name)
    except Exception:
        pass


def get_substage() -> Optional[str]:
    try:
        return _substage_ctx.get()
    except Exception:
        return None


def _get_encoder():
    """Lazily load a tiktoken encoder. gpt-4.1-mini / gpt-4o-mini use o200k_base."""
    global _ENC, _ENC_NAME, _ENC_TRIED
    if _ENC_TRIED:
        return _ENC
    _ENC_TRIED = True
    try:
        import tiktoken  # type: ignore
        for name in ("o200k_base", "cl100k_base"):
            try:
                _ENC = tiktoken.get_encoding(name)
                _ENC_NAME = name
                break
            except Exception:
                continue
    except Exception:
        _ENC = None
    return _ENC


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)  # heuristic fallback (should not happen)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def record(
    *,
    prompt: str,
    completion: str,
    model: str,
    call_site: Optional[str] = None,
    conv_id=None,
    api_usage: Optional[dict] = None,
    latency_s: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one call-level cost row. No-op unless T_MEM_COST_LOG is set."""
    path = _log_path()
    if not path:
        return
    try:
        call_site = call_site or get_substage() or "unknown"
        if conv_id is None:
            conv_id = get_conv()
        if conv_id is not None:
            conv_id = str(conv_id)  # normalise so aggregation keys are consistent

        p_tok = count_tokens(prompt)
        c_tok = count_tokens(completion)

        p_api = c_api = t_api = None
        src = "tiktoken"
        if isinstance(api_usage, dict):
            p_api = api_usage.get("prompt_tokens", api_usage.get("input_tokens"))
            c_api = api_usage.get("completion_tokens", api_usage.get("output_tokens"))
            t_api = api_usage.get("total_tokens")
            if p_api is not None and c_api is not None:
                src = "api"

        if src == "api":
            p_final = int(p_api)
            c_final = int(c_api)
            t_final = int(t_api) if t_api is not None else (p_final + c_final)
        else:
            p_final = p_tok
            c_final = c_tok
            t_final = p_final + c_final

        row = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "pid": os.getpid(),
            "stage": stage_label(),
            "call_site": call_site,
            "conv_id": conv_id,
            "model": model,
            "prompt_tokens": p_final,
            "completion_tokens": c_final,
            "total_tokens": t_final,
            "prompt_tokens_tok": p_tok,
            "completion_tokens_tok": c_tok,
            "prompt_tokens_api": p_api,
            "completion_tokens_api": c_api,
            "total_tokens_api": t_api,
            "token_source": src,
            "enc": _ENC_NAME,
            "prompt_chars": len(prompt or ""),
            "completion_chars": len(completion or ""),
            "latency_s": round(latency_s, 4) if latency_s is not None else None,
        }
        if extra:
            row.update(extra)

        line = json.dumps(row, ensure_ascii=False)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Cost logging must NEVER break the pipeline.
        pass
