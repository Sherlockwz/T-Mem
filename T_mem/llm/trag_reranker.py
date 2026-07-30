"""tRAG bge-reranker-v2-m3 reranker. Same-query batches issue 1 request, otherwise per-pair fallback."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_logger = logging.getLogger("T_mem.trag_reranker")


_TRAG_URL = "http://api.trag.woa.com/v1/trag/retrieval/rerank"
_TRAG_TOKEN = "REDACTED_TMEM_TRAG_TOKEN"
_TRAG_RAG_CODE = "is-42952d3d"
_TRAG_NAMESPACE_CODE = "ns-9128328a"
_TRAG_MODEL = "bge-reranker-v2-m3"


class RerankerProvider:  # pylint: disable=too-few-public-methods
    """bge-reranker-v2-m3 reranker via tRAG."""

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "bge-reranker-v2-m3",
        timeout: int = 120,
        max_retries: int = 10,
        **kwargs,
    ):
        self.base_url = _TRAG_URL
        self.model_name = _TRAG_MODEL
        self.display_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

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
        if not docs:
            return []
        if len(set(queries)) == 1:
            return self._rerank_one_query(queries[0], docs)
        scores = []
        for q, d in zip(queries, docs):
            one = self._rerank_one_query(q, [d])
            scores.append(one[0] if one else 0.0)
        return scores

    def __repr__(self) -> str:
        return f"TragRerankerProvider(display_name={self.display_name})"

    def _rerank_one_query(self, query: str, docs: List[str]) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_TRAG_TOKEN}",
        }
        data = {
            "ragCode": _TRAG_RAG_CODE,
            "namespaceCode": _TRAG_NAMESPACE_CODE,
            "model": _TRAG_MODEL,
            "documents": docs,
            "query": query,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                )
                resp_json = resp.json()
                if isinstance(resp_json, dict) and resp_json.get("data") is not None:
                    sorted_back = sorted(
                        resp_json["data"], key=lambda x: x.get("index", 0)
                    )
                    scores = [float(item.get("relevanceScore", 0.0)) for item in sorted_back]
                    if len(scores) < len(docs):
                        scores += [0.0] * (len(docs) - len(scores))
                    return scores[: len(docs)]
                last_err = f"bad body: {str(resp_json)[:300]}"
                is_rate_limit = (resp_json or {}).get("code") == 429
                _logger.warning(
                    "[tRAG] attempt %d/%d: %s",
                    attempt + 1, self.max_retries, last_err,
                )
            except Exception as e:  # noqa: BLE001
                is_rate_limit = False
                last_err = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[tRAG] attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, last_err,
                )
            if attempt < self.max_retries - 1:
                if is_rate_limit:
                    wait = min(3 * (2 ** attempt) + random.uniform(0, 2), 30)
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)
        _logger.error("[tRAG] all retries exhausted: %s; returning zeros", last_err)
        return [0.0] * len(docs)
