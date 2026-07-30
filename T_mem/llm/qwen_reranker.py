"""Qwen3-reranker provider via Venus llmproxy /v1/rerank.

接口完全兼容 trag_reranker.py 的 RerankerProvider.rerank() 签名，
auth/session/retry 逻辑沿用 bgem3_provider.py 的 Venus 调用模式。

用法 (通过 env var):
    T_MEM_RERANKER_MODEL=qwen3-8b  →  使用 qwen3-reranker-8b (server:282858)
    T_MEM_RERANKER_MODEL=qwen3-6b  →  使用 qwen3-reranker-6b (server:XXXXXX)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_logger = logging.getLogger("T_mem.qwen_reranker")

_VENUS_RERANK_URL = "http://v2.open.venus.oa.com/llmproxy/v1/rerank"

# 模型映射：简写名 → Venus server code
_MODEL_MAP: dict = {
    "qwen3-8b": "server:282858",
    "qwen3-6b": os.environ.get("T_MEM_QWEN3_6B_CODE", "server:XXXXXX"),
}

_DEFAULT_INSTRUCTION = "请根据查询对文档进行相关性排序"


class RerankerProvider:  # pylint: disable=too-few-public-methods
    """Qwen3-reranker provider via Venus llmproxy /v1/rerank.

    bootstrap.py 在启动时根据 T_MEM_RERANKER_MODEL env var 替换
    reranker_provider.RerankerProvider 为此类。
    """

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "qwen3-8b",
        timeout: int = 120,
        max_retries: int = 10,
        **kwargs,
    ):
        self.base_url = _VENUS_RERANK_URL
        # model_name 可能是简写（"qwen3-8b"）或完整 server code
        self.model_code = _MODEL_MAP.get(model_name, model_name)
        self.display_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        # 从 kwargs 或 env 读取 instruction
        self.instruction = kwargs.pop("instruction", None) or os.environ.get(
            "T_MEM_QWEN_RERANK_INSTRUCTION", _DEFAULT_INSTRUCTION
        )

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def rerank(
        self,
        queries: List[str],
        docs: List[str],
        instruction: str | None = None,
    ) -> List[float]:
        """Rerank documents against queries.

        和 trag_reranker 完全一致的调用约定：
        - 所有 query 相同时 → 一次 /v1/rerank 批量调用
        - query 不同时 → 逐对 fallback
        """
        if not docs:
            return []
        if len(set(queries)) == 1:
            return self._rerank_one_query(queries[0], docs, instruction=instruction)
        scores = []
        for q, d in zip(queries, docs):
            one = self._rerank_one_query(q, [d], instruction=instruction)
            scores.append(one[0] if one else 0.0)
        return scores

    def __repr__(self) -> str:
        return f"QwenRerankerProvider(model={self.display_name}, code={self.model_code})"

    def _get_token(self) -> str:
        secret_id = os.environ.get("ENV_VENUS_OPENAPI_SECRET_ID", "")
        app_group = os.environ.get("VENUS_APP_GROUP_ID", "2700")
        return f"{secret_id}@{app_group}"

    def _rerank_one_query(
        self,
        query: str,
        docs: List[str],
        instruction: str | None = None,
    ) -> List[float]:
        if not docs:
            return []

        token = self._get_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        payload: dict = {
            "model": self.model_code,
            "query": query,
            "documents": docs,
        }
        # instruction 字段：优先用函数参数，其次用实例默认值
        instr = instruction if instruction is not None else self.instruction
        if instr:
            payload["instruction"] = instr

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                resp_json = resp.json()

                if resp.status_code == 200:
                    # Venus /v1/rerank 返回 key 是 "results"
                    results = resp_json.get("results") or resp_json.get("data")
                    if isinstance(results, list):
                        sorted_back = sorted(
                            results, key=lambda x: x.get("index", 0)
                        )
                        scores = [
                            float(item.get("relevance_score", 0.0))
                            for item in sorted_back
                        ]
                        if len(scores) < len(docs):
                            scores += [0.0] * (len(docs) - len(scores))
                        return scores[: len(docs)]

                    last_err = f"unexpected body: {str(resp_json)[:300]}"
                else:
                    last_err = f"status={resp.status_code}, body={str(resp_json)[:300]}"

                is_rate_limit = resp.status_code == 429
                _logger.warning(
                    "[qwen-reranker] attempt %d/%d: %s",
                    attempt + 1, self.max_retries, last_err,
                )
            except Exception as e:
                is_rate_limit = False
                last_err = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[qwen-reranker] attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, last_err,
                )

            if attempt < self.max_retries - 1:
                wait = (
                    min(3 * (2 ** attempt) + random.uniform(0, 2), 30)
                    if is_rate_limit
                    else 2 ** attempt + random.uniform(0, 1)
                )
                time.sleep(wait)

        _logger.error(
            "[qwen-reranker] all retries exhausted: %s; returning zeros", last_err
        )
        return [0.0] * len(docs)
