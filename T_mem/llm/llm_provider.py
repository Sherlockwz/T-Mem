"""Generic OpenAI-compatible async LLM provider.

Speaks the standard OpenAI Chat Completions protocol
(``POST {base_url}/chat/completions``) so any OpenAI-compatible endpoint
(OpenAI, vLLM, Ollama, SiliconFlow, LM Studio, ...) can be used as-is.

Configuration (all via environment variables, see ``.env.example``):

- ``T_MEM_LLM_BASE_URL``  base URL of the OpenAI-compatible endpoint
                          (default: ``https://api.openai.com/v1``)
- ``T_MEM_LLM_API_KEY``   API key (fallback: ``OPENAI_API_KEY``)
- ``T_MEM_MAX_CONCURRENCY`` global cap on in-flight LLM requests
- ``T_MEM_RETRIES``       network retries per call (default 4)
- ``T_MEM_JSON_MAX_RETRIES`` re-asks for valid JSON (default 3)
- ``T_MEM_TIMEOUT``       per-request timeout in seconds (default 420)
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:  # pragma: no cover
    import json as _json

    class _FallbackJsonRepair:
        @staticmethod
        def loads(s):
            return _json.loads(s)

    json_repair = _FallbackJsonRepair()
    HAS_JSON_REPAIR = False

import json as _json
import logging
import os
import random

import requests

_logger = logging.getLogger("T_mem.llm")


def _base_url() -> str:
    """OpenAI-compatible endpoint base URL (no trailing slash)."""
    return os.environ.get("T_MEM_LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")


def _api_key() -> str:
    """API key for the OpenAI-compatible endpoint."""
    return os.environ.get("T_MEM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def _endpoint() -> str:
    return _base_url() + "/chat/completions"


# Substrings that indicate a retryable upstream error. The last entry is a
# Chinese upstream error message emitted by some hosted providers.
_RECOVERABLE_ERRORS = (
    "Overload",
    "Too many",
    "overloaded",
    "rate limit",
    "rate_limit",
    "unable to process your request",
    "模型出错了，请稍后重试",
)


def _extract_usage(ret) -> Optional[dict]:
    """Best-effort extraction of an OpenAI-style usage dict from a response."""
    if not isinstance(ret, dict):
        return None
    data = ret.get("data")
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        return data["usage"]
    if isinstance(ret.get("usage"), dict):
        return ret["usage"]
    return None


def _record_cost_safe(prompt, response, model, ret, meta, latency_s) -> None:
    """Append one cost-ledger row for a successful completion. Never raises."""
    try:
        from T_mem.utils.cost_ledger import enabled, record
        if not enabled():
            return
        call_site = meta.get("call_site") if isinstance(meta, dict) else None
        conv_id = meta.get("conv_id") if isinstance(meta, dict) else None
        record(
            prompt=prompt,
            completion=response or "",
            model=model,
            call_site=call_site,
            conv_id=conv_id,
            api_usage=_extract_usage(ret),
            latency_s=latency_s,
        )
    except Exception:
        pass


def chat_completion(prompt: str, model: str, timeout: int = 420, max_retries: int = 4,
                    temperature: float = 0.0, meta: Optional[dict] = None) -> str:
    """Single-turn OpenAI-compatible chat/completions call with retries.

    Returns the raw completion text. Raises ``RuntimeError`` after all
    retries are exhausted. The response is recorded into the cost ledger.
    """
    import hashlib as _hashlib
    url = _endpoint()
    key = _api_key()
    prompt_sha = _hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:8]
    prompt_len = len(prompt)
    approx_tokens = prompt_len // 4

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        backoff = random.uniform(0.2, 0.8)
        _call_start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, data=_json.dumps(body), timeout=timeout)
            ret = resp.json()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            _logger.warning(
                "[LLM] request exception attempt %d/%d: %s: %s, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                attempt, max_retries, type(e).__name__, e, model, prompt_sha, prompt_len, approx_tokens, backoff,
            )
            time.sleep(backoff)
            continue

        # Non-2xx responses (401/429/5xx ...) are retried.
        if not resp.ok:
            is_rate_limit = resp.status_code in (429,) or "rate" in str(ret).lower()
            last_exc = RuntimeError(
                f"HTTP {resp.status_code} from {url}, model={model}, body={str(ret)[:200]}"
            )
            _logger.warning(
                "[LLM] non-2xx attempt %d/%d: %s, sleeping %.2fs",
                attempt, max_retries, last_exc, backoff,
            )
            if is_rate_limit and attempt < max_retries:
                rl_base = float(os.environ.get("T_MEM_RATELIMIT_BACKOFF", "12"))
                time.sleep(rl_base + random.uniform(0.0, rl_base * 0.5))
            else:
                time.sleep(backoff)
            continue

        if not isinstance(ret, dict) or not isinstance(ret.get("choices"), list):
            last_exc = RuntimeError(f"malformed upstream response, model={model}, ret={ret!r}")
            _logger.warning(
                "[LLM] malformed response attempt %d/%d, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                attempt, max_retries, model, prompt_sha, prompt_len, approx_tokens, backoff,
            )
            time.sleep(backoff)
            continue

        try:
            response = ret["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last_exc = RuntimeError(f"no choices[0].message.content, model={model}, ret={ret!r}")
            _logger.warning(
                "[LLM] missing content attempt %d/%d, model=%s, sha=%s, sleeping %.2fs",
                attempt, max_retries, model, prompt_sha, backoff,
            )
            time.sleep(backoff)
            continue

        if response is None or str(response).strip() == "":
            last_exc = RuntimeError(f"empty response, model={model}")
            _logger.warning(
                "[LLM] empty response attempt %d/%d, model=%s, sha=%s, sleeping %.2fs",
                attempt, max_retries, model, prompt_sha, backoff,
            )
            time.sleep(backoff)
            continue

        if any(err in str(response) for err in _RECOVERABLE_ERRORS):
            last_exc = RuntimeError(
                f"recoverable upstream error, model={model}, resp={str(response)[:200]}"
            )
            _logger.warning(
                "[LLM] recoverable upstream error attempt %d/%d, backoff %.2fs, model=%s, sha=%s, len=%d, resp=%s",
                attempt, max_retries, backoff, model, prompt_sha, prompt_len, str(response)[:120],
            )
            time.sleep(backoff)
            continue

        if attempt > 1:
            _logger.warning(
                "[LLM] sha=%s recovered at attempt %d/%d (len=%d, ~%dtok, model=%s)",
                prompt_sha, attempt, max_retries, prompt_len, approx_tokens, model,
            )
        _record_cost_safe(
            prompt, response, model, ret, meta,
            time.perf_counter() - _call_start,
        )
        return response

    raise RuntimeError(
        f"LLM chat failed after {max_retries} retries, "
        f"model={model}, sha={prompt_sha}, len={prompt_len}, "
        f"endpoint={url}, last_error={last_exc}"
    )


class LLMProvider:
    """Async OpenAI-compatible LLM provider with optional JSON re-asking.

    Keeps the same ``generate()`` contract as previous internal providers so
    all pipeline stages can be switched without changes.
    """

    _shared_executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    def __init__(
        self,
        provider_type: str = "openai",
        model: Optional[str] = None,
        timeout: int = 420,
        retries: int = 4,
        json_max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 16384,
        max_workers: int = 14,
        enable_stats: bool = False,
        failure_logger: Any = None,
        **kwargs,
    ):
        # Global concurrency cap: env override lets us throttle ALL build/QA
        # stages from a single knob.
        try:
            _mw_env = int(
                os.environ.get("T_MEM_MAX_CONCURRENCY")
                or str(max_workers)
            )
            if _mw_env > 0:
                max_workers = _mw_env
        except Exception:
            pass

        self.model = model
        self.timeout = int(os.environ.get("T_MEM_TIMEOUT", str(timeout)))
        self.retries = max(1, int(os.environ.get("T_MEM_RETRIES", str(retries))))
        self.json_max_retries = max(1, int(os.environ.get("T_MEM_JSON_MAX_RETRIES", str(json_max_retries))))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats
        self.failure_logger = failure_logger
        self._extra_kwargs = kwargs

        with self._executor_lock:
            if LLMProvider._shared_executor is None:
                LLMProvider._shared_executor = ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="tmem-llm"
                )
        self._executor = LLMProvider._shared_executor

        self._stats_lock = threading.Lock()
        self.accumulated_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "total_duration": 0.0,
        }

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> str:
        """Generate a response; optionally re-ask the same model for valid JSON."""
        want_json = bool(
            response_format and response_format.get("type") == "json_object"
        )
        call_site = (kwargs.get("call_site") or self._extra_kwargs.get("call_site")
                     or "unknown")
        conv_id = kwargs.get("conv_id")
        if conv_id is None:
            # Read the per-conversation contextvar in the async task context
            # (executor threads cannot see it), then thread it down via `meta`.
            try:
                from T_mem.utils.cost_ledger import get_conv
                conv_id = get_conv()
            except Exception:
                conv_id = None
        _cost_meta = {"call_site": call_site, "conv_id": conv_id}

        used_temp = temperature if temperature is not None else self.temperature

        effective_prompt = prompt
        if want_json:
            effective_prompt = (
                "IMPORTANT: Your response MUST be a single valid JSON object and nothing else. "
                "Do NOT wrap it in markdown code fences. Do NOT add any prose before or after.\n\n"
                + prompt
            )

        attempts_budget = self.json_max_retries if want_json else 1
        last_err: Optional[str] = None
        last_raw: Optional[str] = None
        total_attempts = 0

        for attempt in range(1, attempts_budget + 1):
            total_attempts += 1
            start = time.perf_counter()
            loop = asyncio.get_running_loop()
            try:
                raw = await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        chat_completion,
                        effective_prompt,
                        self.model,
                        self.timeout,
                        self.retries,
                        float(used_temp),
                        _cost_meta,
                    ),
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                last_raw = None
                _logger.warning(
                    "[LLM] generate exception attempt %d/%d at %s (model=%s): %s",
                    attempt, attempts_budget, call_site, self.model, last_err[:200],
                )
                if attempt < attempts_budget:
                    await asyncio.sleep(0.5)
                    continue
                break

            duration = time.perf_counter() - start
            self._record_stats(duration)

            if not want_json:
                return raw

            parsed, parse_err = self._safe_parse_json(raw)
            if parsed is not None:
                return _json.dumps(parsed, ensure_ascii=False)

            last_err = parse_err
            last_raw = raw
            _logger.warning(
                "[LLM] JSON parse failed attempt %d/%d at %s (model=%s): %s",
                attempt, attempts_budget, call_site, self.model,
                parse_err[:200] if parse_err else "",
            )
            if attempt < attempts_budget:
                await asyncio.sleep(0.5)

        self._log_failure(conv_id, call_site, total_attempts, last_err or "",
                          prompt, last_raw or "")
        raise RuntimeError(
            f"LLM generate failed after {total_attempts} attempts at "
            f"{call_site} against model={self.model}: {last_err}"
        )

    async def test_connection(self) -> bool:
        try:
            resp = await self.generate("Say 'hello' as a short plain word.")
            return bool(resp)
        except Exception as e:
            _logger.error("test_connection failed: %s", e)
            return False

    def get_accumulated_stats(self) -> Optional[dict]:
        with self._stats_lock:
            return dict(self.accumulated_stats)

    def reset_accumulated_stats(self) -> None:
        with self._stats_lock:
            for k in self.accumulated_stats:
                self.accumulated_stats[k] = 0 if k != "total_duration" else 0.0

    def __repr__(self) -> str:
        return f"LLMProvider(model={self.model})"

    @staticmethod
    def _safe_parse_json(raw: str):
        if raw is None:
            return None, "empty response"
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            first_nl = cleaned.find("\n")
            if first_nl >= 0 and (cleaned[:first_nl].strip().lower() in ("json", "")):
                cleaned = cleaned[first_nl + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3]
            cleaned = cleaned.strip()
        try:
            obj = json_repair.loads(cleaned)
            if obj is None or obj == "":
                return None, "json_repair returned empty"
            return obj, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _record_stats(self, duration: float) -> None:
        with self._stats_lock:
            self.accumulated_stats["call_count"] += 1
            self.accumulated_stats["total_duration"] += duration

    def _log_failure(self, conv_id, call_site, attempt_count, last_error,
                     prompt, raw_response):
        if self.failure_logger is None:
            return
        try:
            self.failure_logger.log(
                conv_id=conv_id,
                call_site=call_site,
                attempt_count=attempt_count,
                last_error=last_error,
                prompt_preview=prompt[:800] if prompt else "",
                raw_response=raw_response[:2000] if raw_response else "",
                model=self.model,
            )
        except Exception as e:
            _logger.error("failure_logger.log itself failed: %s", e)


from datetime import datetime as _datetime  # noqa: E402


class JsonFailureLogger:
    """Thread-safe append-only JSONL writer for JSON-parse failures."""

    def __init__(self, log_path: str):
        self.log_path = str(log_path)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        conv_id: Any = None,
        call_site: str = "unknown",
        attempt_count: int = 0,
        last_error: str = "",
        prompt_preview: str = "",
        raw_response: str = "",
        model: str = "",
    ) -> None:
        entry = {
            "timestamp": _datetime.now().isoformat(timespec="seconds"),
            "conv_id": conv_id,
            "call_site": call_site,
            "model": model,
            "attempt_count": int(attempt_count),
            "last_error": str(last_error)[:2000],
            "prompt_preview": (prompt_preview or "")[:1200],
            "raw_response": (raw_response or "")[:2000],
        }
        line = _json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def __repr__(self) -> str:
        return f"JsonFailureLogger(log_path={self.log_path})"


_GLOBAL_FAILURE_LOGGER = None


def set_global_failure_logger(logger_obj) -> None:
    global _GLOBAL_FAILURE_LOGGER
    _GLOBAL_FAILURE_LOGGER = logger_obj


def get_global_failure_logger():
    return _GLOBAL_FAILURE_LOGGER
