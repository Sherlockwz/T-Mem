"""BGE-M3 embedding provider via Venus llmproxy (OpenAI-compatible /embeddings endpoint)."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import List

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_logger = logging.getLogger("T_mem.bgem3_embedding")

_VENUS_EMBEDDING_URL = "http://v2.open.venus.oa.com/llmproxy/embeddings"
_BGE_M3_MODEL_CODE = "server:269176"


class EmbeddingProvider:  # pylint: disable=too-few-public-methods
    """BGE-M3 embedding provider served via Venus llmproxy."""

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "bge-m3",
        timeout: int = 120,
        max_retries: int = 10,
        **kwargs,
    ):
        self.base_url = _VENUS_EMBEDDING_URL
        self.model_name = _BGE_M3_MODEL_CODE
        self.display_name = model_name or "bge-m3"
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

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        secret_id = os.environ.get("ENV_VENUS_OPENAPI_SECRET_ID", "")
        _app_group = os.environ.get("VENUS_APP_GROUP_ID", "2700")
        token = f"{secret_id}@{_app_group}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        data = {"model": self.model_name, "input": texts}

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
                if (
                    isinstance(resp_json, dict)
                    and isinstance(resp_json.get("data"), list)
                    and resp_json["data"]
                ):
                    sorted_data = sorted(
                        resp_json["data"], key=lambda x: x.get("index", 0)
                    )
                    return [item["embedding"] for item in sorted_data]
                is_rate_limit = resp.status_code == 429
                last_err = f"bad body: {str(resp_json)[:300]}"
                _logger.warning(
                    "[BGE-M3] attempt %d/%d: %s (status=%s)",
                    attempt + 1, self.max_retries, last_err, resp.status_code,
                )
            except Exception as e:  # noqa: BLE001
                is_rate_limit = False
                last_err = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[BGE-M3] attempt %d/%d request failed: %s",
                    attempt + 1, self.max_retries, last_err,
                )

            if attempt < self.max_retries - 1:
                if is_rate_limit:
                    wait = min(3 * (2 ** attempt) + random.uniform(0, 2), 30)
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)

        raise RuntimeError(
            f"BGE-M3 embed failed after {self.max_retries} retries: {last_err}"
        )

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        dot_product = np.dot(doc_vecs, query_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        denominator = query_norm * doc_norms
        denominator = np.where(denominator == 0, 1e-9, denominator)
        return dot_product / denominator

    def __repr__(self) -> str:
        return f"BGEM3EmbeddingProvider(display_name={self.display_name})"
