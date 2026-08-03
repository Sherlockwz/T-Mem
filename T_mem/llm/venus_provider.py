"""Async LLM provider backed by Venus chat/single. Exposes venus_chat / VenusLLMProvider / LLMProvider shim."""

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

_logger = logging.getLogger("T_mem.venus_llm")


_SECRET_ID = os.environ.get("ENV_VENUS_OPENAPI_SECRET_ID")
_SECRET_KEY = os.environ.get("ENV_VENUS_OPENAPI_SECRET_KEY")

_VENUS_CLIENT = None
_VENUS_CLIENT_CACHE: dict = {}
_VENUS_CLIENT_LOCK = threading.Lock()


def _get_venus_client(timeout: int):
    global _VENUS_CLIENT_CACHE
    with _VENUS_CLIENT_LOCK:
        if timeout not in _VENUS_CLIENT_CACHE:
            from venus_api_base.http_client import HttpClient
            from venus_api_base.config import Config
            _VENUS_CLIENT_CACHE[timeout] = HttpClient(
                config=Config(read_timeout=timeout),
                secret_id=_SECRET_ID,
                secret_key=_SECRET_KEY,
            )
        return _VENUS_CLIENT_CACHE[timeout]


_RECOVERABLE_ERRORS = (
    "未知错误",
    "Overload",
    "Too many",
    "模型出错了，请稍后重试",
    "unable to process your request",
)


def _parse_venus_server_id(model: str) -> Optional[int]:
    """Return a Venus serverId for explicit server-backed models.

    Accepted spellings are intentionally boring and shell-friendly:
    `venus-server-284227`, `server:284227`, `server-284227`, or just `284227`.
    Plain model aliases such as `gpt-4o-mini` still go through the standard
    Venus `model` field.
    """
    raw = str(model or "").strip()
    low = raw.lower()
    for prefix in ("venus-server-", "venus_server_", "server:", "server-"):
        if low.startswith(prefix):
            tail = raw[len(prefix):].strip()
            return int(tail) if tail.isdigit() else None
    return int(raw) if raw.isdigit() else None


def _extract_usage(ret) -> Optional[dict]:
    """Best-effort extraction of an OpenAI-style usage dict from a Venus response."""
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


def venus_chat(prompt: str, model: str, timeout: int = 420, max_retries: int = 4,
               temperature: float = 0.0, meta: Optional[dict] = None) -> str:
    """Single-turn Venus chat/single call with same-model retries."""
    # CRITICAL: do NOT pass `do_sample` here — OpenAI-family upstreams reject the
    # kwarg and may silently return an empty body.
    client = _get_venus_client(timeout)
    header = {"Content-Type": "application/json"}
    last_exc: Optional[BaseException] = None
    server_id = _parse_venus_server_id(model)

    import hashlib as _hashlib
    import requests as _requests
    prompt_sha = _hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:8]
    prompt_len = len(prompt)
    approx_tokens = prompt_len // 4

    active_model = model

    for attempt in range(1, max_retries + 1):
        if server_id is None:
            body = {
                "appGroupId": int(os.environ.get("VENUS_APP_GROUP_ID", "2700")),
                "model": active_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
        else:
            body = {
                "appGroupId": int(os.environ.get("T_MEM_VENUS_SERVER_APP_GROUP_ID", "2700")),
                "serverId": server_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": float(os.environ.get("T_MEM_VENUS_SERVER_TOP_P", "1")),
                "top_k": int(os.environ.get("T_MEM_VENUS_SERVER_TOP_K", "1")),
                "do_sample": False,
                "num_beams": 1,
                "max_length": int(os.environ.get("T_MEM_VENUS_SERVER_MAX_LENGTH", "16384")),
                "max_new_tokens": int(os.environ.get("T_MEM_VENUS_SERVER_MAX_NEW_TOKENS", "4096")),
                "length_penalty": 1.0,
                "repetition_penalty": float(os.environ.get("T_MEM_VENUS_SERVER_REPETITION_PENALTY", "1.0")),
                "stream": False,
                "delete_prompt_from_output": 1,
                "n": 1,
                "task_id": time.time(),
            }

        backoff = random.uniform(0.2, 0.8)
        _call_start = time.perf_counter()
        try:
            if server_id is None:
                ret = client.post(
                    "http://v2.open.venus.oa.com/chat/single",
                    header=header,
                    body=_json.dumps(body),
                )
            else:
                path = "/chat/single"
                payload = _json.dumps(body)
                timestamp = str(int(time.time()))
                sign = client.gen_venus_sign_with_timestamp(path, "", payload, timestamp)
                headers = {
                    "Content-Type": "application/json",
                    "Venusopenapi-Request-Timestamp": timestamp,
                    "Venusopenapi-Authorization": sign,
                    "Venusopenapi-Secret-Id": _SECRET_ID,
                }
                resp = _requests.post(
                    "http://v2.open.venus.oa.com/chat/single",
                    headers=headers,
                    data=payload,
                    timeout=timeout,
                )
                ret = resp.json()
            if not isinstance(ret, dict) or not isinstance(ret.get("data"), dict):
                # Detect 4029 public-service RATE-LIMIT ("并发有限; 限流为: 50/min").
                # It is NOT a malformed body — it is a throttle. Back off LONG so
                # the retry lands in the next quota window instead of hammering.
                is_rate_limit = False
                if isinstance(ret, dict):
                    _code = ret.get("code") or ret.get("retCode")
                    _msg = str(ret.get("message") or ret.get("msg") or "")
                    if _code == 4029 or "限流" in _msg or "并发有限" in _msg:
                        is_rate_limit = True
                if is_rate_limit:
                    rl_base = float(os.environ.get("T_MEM_RATELIMIT_BACKOFF", "12"))
                    rl_backoff = rl_base + random.uniform(0.0, rl_base * 0.5)
                    _logger.warning(
                        "[VenusLLM] RATE-LIMIT 4029 attempt %d/%d, model=%s, sha=%s, len=%d (~%dtok), sleeping %.1fs (quota refresh)",
                        attempt, max_retries, active_model, prompt_sha, prompt_len, approx_tokens, rl_backoff,
                    )
                    last_exc = RuntimeError(
                        f"Venus 4029 rate-limit, model={active_model}"
                    )
                    time.sleep(rl_backoff)
                    continue
                _logger.warning(
                    "[VenusLLM] malformed upstream response attempt %d/%d, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                    attempt, max_retries, active_model, prompt_sha, prompt_len, approx_tokens, backoff,
                )
                last_exc = RuntimeError(
                    f"Venus malformed upstream response, model={active_model}, ret={ret!r}"
                )
                time.sleep(backoff)
                continue
            response = ret["data"].get("response")
            if response is None:
                _logger.warning(
                    "[VenusLLM] None response attempt %d/%d, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                    attempt, max_retries, active_model, prompt_sha, prompt_len, approx_tokens, backoff,
                )
                last_exc = RuntimeError(
                    f"Venus None response, model={active_model}"
                )
                time.sleep(backoff)
                continue
            if any(err in response for err in _RECOVERABLE_ERRORS):
                _logger.warning(
                    "[VenusLLM] recoverable upstream error attempt %d/%d, backoff %.2fs, model=%s, sha=%s, len=%d, resp=%s",
                    attempt, max_retries, backoff, active_model, prompt_sha, prompt_len, response[:120],
                )
                last_exc = RuntimeError(
                    f"Venus recoverable upstream error, model={active_model}, resp={response[:200]}"
                )
                time.sleep(backoff)
                continue
            if attempt > 1:
                _logger.warning(
                    "[VenusLLM] sha=%s recovered at attempt %d/%d (len=%d, ~%dtok, model=%s)",
                    prompt_sha, attempt, max_retries, prompt_len, approx_tokens,
                    active_model,
                )
            _record_cost_safe(
                prompt, response, active_model, ret, meta,
                time.perf_counter() - _call_start,
            )
            return response
        except AttributeError as e:
            last_exc = e
            _logger.warning(
                "[VenusLLM] AttributeError (likely None upstream) attempt %d/%d: %s, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                attempt, max_retries, e, active_model, prompt_sha, prompt_len, approx_tokens, backoff,
            )
            if attempt == max_retries:
                try:
                    diag_path = os.environ.get(
                        "T_MEM_VENUS_DIAG_DIR",
                        os.path.dirname(os.environ.get("T_MEM_FAILURE_LOG", "/tmp/venus_diag")),
                    )
                    os.makedirs(diag_path, exist_ok=True)
                    diag_file = os.path.join(diag_path, f"venus_stuck_{prompt_sha}.txt")
                    with open(diag_file, "w", encoding="utf-8") as f:
                        f.write(f"sha={prompt_sha}\nmodel={model}\nactive_model={active_model}\nlen={prompt_len}\napprox_tokens={approx_tokens}\n")
                        f.write(f"first_err={e!r}\n")
                        f.write("---PROMPT---\n")
                        f.write(prompt)
                    _logger.warning(
                        "[VenusLLM] DUMPED stuck prompt sha=%s to %s", prompt_sha, diag_file,
                    )
                except Exception as dump_err:
                    _logger.warning("[VenusLLM] failed to dump stuck prompt: %s", dump_err)
            time.sleep(backoff)
        except Exception as e:
            last_exc = e
            _logger.warning(
                "[VenusLLM] exception attempt %d/%d: %s: %s, model=%s, sha=%s, len=%d (~%dtok), sleeping %.2fs",
                attempt, max_retries, type(e).__name__, e, active_model, prompt_sha, prompt_len, approx_tokens, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Venus chat failed after {max_retries} retries (strict single-model), "
        f"model={model}, sha={prompt_sha}, len={prompt_len}, "
        f"last_error={last_exc}"
    )


class VenusLLMProvider:
    """Async LLM provider backed by Venus chat/single."""

    _shared_executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    def __init__(
        self,
        model: str,
        timeout: int = 420,
        venus_retries: int = 4,
        json_max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 16384,
        max_workers: int = 14,
        enable_stats: bool = False,
        failure_logger: Any = None,
        **kwargs,
    ):
        # Global concurrency cap: env override lets us throttle ALL build/QA stages
        # to Venus from a single knob (lower = fewer malformed/timeout under load).
        try:
            _mw_env = int(os.environ.get("T_MEM_VENUS_MAX_WORKERS", "").strip() or max_workers)
            if _mw_env > 0:
                max_workers = _mw_env
        except Exception:
            pass
        self.model = model
        self.timeout = timeout
        self.venus_retries = max(1, venus_retries)
        self.json_max_retries = max(1, json_max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats
        self.failure_logger = failure_logger

        self._extra_kwargs = kwargs

        with self._executor_lock:
            if VenusLLMProvider._shared_executor is None:
                VenusLLMProvider._shared_executor = ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="venus-llm"
                )
        self._executor = VenusLLMProvider._shared_executor

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
                        venus_chat,
                        effective_prompt,
                        self.model,
                        self.timeout,
                        self.venus_retries,
                        float(used_temp),
                        _cost_meta,
                    ),
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                last_raw = None
                _logger.warning(
                    "[VenusLLM] generate exception attempt %d/%d at %s (model=%s): %s",
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
                "[VenusLLM] JSON parse failed attempt %d/%d at %s (model=%s): %s",
                attempt, attempts_budget, call_site, self.model,
                parse_err[:200] if parse_err else "",
            )
            if attempt < attempts_budget:
                await asyncio.sleep(0.5)

        self._log_failure(conv_id, call_site, total_attempts, last_err or "",
                           prompt, last_raw or "")
        raise RuntimeError(
            f"VenusLLM generate failed after {total_attempts} attempts at "
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
        return f"VenusLLMProvider(model={self.model})"

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
        if not self.enable_stats:
            with self._stats_lock:
                self.accumulated_stats["call_count"] += 1
                self.accumulated_stats["total_duration"] += duration
            return
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


class PolarisLLMProvider:
    """LLM provider for the teacher's locally-deployed model via the Polaris
    freeform endpoint (used by dan_memory/core/utils.py::get_hunyuan_result).
    Used for building the memory library with llama-3.1-8b ("llama-8b") which is
    NOT served by Venus. Same async `generate()` contract as VenusLLMProvider.
    Requires network route to *.turbotke.production.polaris (jumpbox: `go -g ...`).
    """
    _shared_executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    _WSID = "11685"
    _URL = f"http://stream-server-online-sbs-{_WSID}.turbotke.production.polaris:81/openapi/chat/completions_freeform"
    _AUTH = "Bearer REDACTED_TMEM_POLARIS_TOKEN"

    def __init__(self, model: str, temperature: float = 0.0,
                 json_max_retries: int = 3, max_workers: int = 8,
                 failure_logger: Any = None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.json_max_retries = max(1, json_max_retries)
        self.failure_logger = failure_logger
        self.net_retries = int(os.environ.get("T_MEM_POLARIS_RETRIES", "6"))
        self.timeout = int(os.environ.get("T_MEM_POLARIS_TIMEOUT", "180"))
        self.accumulated_stats = {"call_count": 0, "total_duration": 0.0}
        self._stats_lock = threading.Lock()
        with self._executor_lock:
            if PolarisLLMProvider._shared_executor is None:
                _mw = int(os.environ.get("T_MEM_VENUS_MAX_WORKERS", "").strip() or max_workers)
                PolarisLLMProvider._shared_executor = ThreadPoolExecutor(
                    max_workers=max(1, _mw), thread_name_prefix="polaris-llm")
        self._executor = PolarisLLMProvider._shared_executor

    def _blocking_call(self, prompt: str, temperature: float) -> str:
        import requests as _requests
        headers = {"Content-Type": "application/json", "Authorization": self._AUTH, "Wsid": self._WSID}
        body = {
            "model": self.model, "query_id": str(time.time()),
            "enable_sse": True, "openai_infer": True, "stream": False, "do_sample": False,
            "messages": [{"role": "user", "content": prompt}],
            "beam_size": "1", "temperature": float(temperature), "top_p": 0.6, "top_k": 1,
            "repetition_penalty": 1.1, "output_seq_len": 4096, "max_input_seq_len": 16384,
            "random_seed": 1234, "domain_index": -1, "decoupled": "1",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last = None
        for _ in range(self.net_retries):
            try:
                resp = _requests.post(self._URL, headers=headers, json=body,
                                      stream=False, timeout=self.timeout)
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last = e
                time.sleep(2)
                continue
        raise RuntimeError(f"Polaris endpoint failed after {self.net_retries} retries, "
                           f"model={self.model}, last={last}")

    async def generate(self, prompt: str, temperature: float | None = None,
                       extra_body: dict | None = None,
                       response_format: dict | None = None, **kwargs) -> str:
        want_json = bool(response_format and response_format.get("type") == "json_object")
        used_temp = temperature if temperature is not None else self.temperature
        eff = prompt
        if want_json:
            eff = ("IMPORTANT: Your response MUST be a single valid JSON object and nothing else. "
                   "Do NOT wrap it in markdown code fences. Do NOT add any prose before or after.\n\n" + prompt)
        budget = self.json_max_retries if want_json else 1
        last_err = None; last_raw = None
        for attempt in range(1, budget + 1):
            loop = asyncio.get_running_loop()
            start = time.perf_counter()
            try:
                raw = await loop.run_in_executor(
                    self._executor, functools.partial(self._blocking_call, eff, float(used_temp)))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"; last_raw = None
                if attempt < budget:
                    await asyncio.sleep(0.5); continue
                break
            with self._stats_lock:
                self.accumulated_stats["call_count"] += 1
                self.accumulated_stats["total_duration"] += time.perf_counter() - start
            if not want_json:
                return raw
            parsed, perr = VenusLLMProvider._safe_parse_json(raw)
            if parsed is not None:
                return _json.dumps(parsed, ensure_ascii=False)
            last_err = perr; last_raw = raw
            if attempt < budget:
                await asyncio.sleep(0.5)
        if self.failure_logger is not None:
            try:
                self.failure_logger.log(conv_id=kwargs.get("conv_id"),
                    call_site=kwargs.get("call_site", "unknown"), attempt_count=budget,
                    last_error=last_err or "", prompt_preview=prompt[:800],
                    raw_response=(last_raw or "")[:2000], model=self.model)
            except Exception:
                pass
        raise RuntimeError(f"Polaris generate failed after {budget} attempts, "
                           f"model={self.model}: {last_err}")

    async def test_connection(self) -> bool:
        try:
            return bool(await self.generate("Say hello."))
        except Exception:
            return False

    def get_accumulated_stats(self):
        with self._stats_lock:
            return dict(self.accumulated_stats)

    def reset_accumulated_stats(self):
        with self._stats_lock:
            self.accumulated_stats = {"call_count": 0, "total_duration": 0.0}


def _is_polaris_model(model: str) -> bool:
    """Models that must be routed to the Polaris endpoint (not Venus)."""
    m = (model or "").lower()
    explicit = os.environ.get("T_MEM_POLARIS_MODELS", "").strip().lower()
    names = {x.strip() for x in explicit.split(",") if x.strip()} or {"llama-8b", "llama-3.1-8b"}
    return m in names or m.startswith("llama")


class LLMProvider:  # pylint: disable=too-few-public-methods
    """Common-name wrapper around VenusLLMProvider (so bootstrap can monkey-patch it in)."""

    def __init__(self, provider_type: str = "openai", **kwargs):
        self.provider_type = provider_type
        model = kwargs.pop("model", None)
        if not model:
            raise ValueError(
                "LLMProvider requires an explicit 'model' kwarg; "
                "model ids are declared in T_mem.config.MODELS."
            )
        # Route llama (locally-deployed, NOT on Venus) to the Polaris endpoint.
        if _is_polaris_model(model):
            temperature = kwargs.pop("temperature", 0.0)
            kwargs.pop("max_tokens", None); kwargs.pop("enable_stats", None)
            kwargs.pop("base_url", None); kwargs.pop("api_key", None); kwargs.pop("llm_provider", None)
            json_max_retries = int(os.environ.get("T_MEM_JSON_MAX_RETRIES", "3"))
            self.provider = PolarisLLMProvider(
                model=model, temperature=temperature,
                json_max_retries=json_max_retries,
                failure_logger=get_global_failure_logger(), **kwargs)
            return
        temperature = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", 16384)
        enable_stats = kwargs.pop("enable_stats", False)
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        kwargs.pop("llm_provider", None)

        venus_retries = int(os.environ.get("T_MEM_VENUS_RETRIES", "3"))
        json_max_retries = int(os.environ.get("T_MEM_JSON_MAX_RETRIES", "3"))
        venus_timeout = int(os.environ.get("T_MEM_VENUS_TIMEOUT", "420"))
        failure_logger = get_global_failure_logger()

        self.provider = VenusLLMProvider(
            model=model,
            timeout=venus_timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_stats=enable_stats,
            venus_retries=venus_retries,
            json_max_retries=json_max_retries,
            failure_logger=failure_logger,
            **kwargs,
        )

    async def generate(self, prompt: str, temperature: float | None = None,
                        extra_body: dict | None = None,
                        response_format: dict | None = None,
                        **kwargs) -> str:
        return await self.provider.generate(
            prompt,
            temperature=temperature,
            extra_body=extra_body,
            response_format=response_format,
            **kwargs,
        )

    async def test_connection(self) -> bool:
        return await self.provider.test_connection()

    def get_accumulated_stats(self):
        return self.provider.get_accumulated_stats()

    def reset_accumulated_stats(self):
        return self.provider.reset_accumulated_stats()

    def __repr__(self) -> str:
        return f"LLMProvider({self.provider!r})"


from datetime import datetime as _datetime


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
