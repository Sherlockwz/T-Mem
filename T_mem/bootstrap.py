"""Bootstrap: monkey-patch placeholder provider modules with real ones (Venus + BGE-M3 + tRAG).
Must be imported BEFORE any provider module is referenced.
Env: T_MEM_EXPERIMENT_NAME / RESULTS_DIR / DATA_FILE / NUM_CONV / USE_RERANKER."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger("T_mem.bootstrap")

_PATCHED = False


def patch_providers():
    """Idempotently replace placeholder providers with Venus-backed ones."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from T_mem import config as hm_config

    custom_results_dir = os.environ.get("T_MEM_RESULTS_DIR", "").strip()
    if custom_results_dir:
        hm_config.RESULTS_DIR = Path(custom_results_dir).resolve()
        _logger.info("[bootstrap] RESULTS_DIR overridden -> %s", hm_config.RESULTS_DIR)

    custom_dataset = os.environ.get("T_MEM_DATA_FILE", "").strip()
    if custom_dataset:
        hm_config.ExperimentConfig.dataset_path = str(Path(custom_dataset).resolve())
        _logger.info("[bootstrap] dataset_path overridden -> %s",
                      hm_config.ExperimentConfig.dataset_path)

    custom_num_conv = os.environ.get("T_MEM_NUM_CONV", "").strip()
    if custom_num_conv:
        try:
            hm_config.ExperimentConfig.num_conv = int(custom_num_conv)
            _logger.info("[bootstrap] num_conv overridden -> %d",
                          hm_config.ExperimentConfig.num_conv)
        except ValueError:
            _logger.warning("[bootstrap] T_MEM_NUM_CONV=%r is not int, ignored",
                             custom_num_conv)

    env_rr = os.environ.get("T_MEM_USE_RERANKER", "").strip().lower()
    if env_rr in {"0", "false", "no", "off"}:
        hm_config.ExperimentConfig.use_reranker = False
        _logger.info("[bootstrap] use_reranker EXPLICITLY disabled by env")
    else:
        _rr_model = os.environ.get("T_MEM_RERANKER_MODEL", "").strip()
        if _rr_model:
            _logger.info("[bootstrap] use_reranker ENABLED (model=%s)", _rr_model)
        else:
            _logger.info("[bootstrap] use_reranker ENABLED (bge-reranker-v2-m3 via tRAG)")

    _logger.info(
        "[bootstrap] experiment_name = %s (no suffix; equals T_MEM_EXPERIMENT_NAME)",
        hm_config.ExperimentConfig.experiment_name,
    )

    log_path = os.environ.get("T_MEM_FAILURE_LOG", "").strip()
    if log_path:
        from T_mem.llm.venus_provider import JsonFailureLogger, set_global_failure_logger
        failure_logger = JsonFailureLogger(log_path)
        set_global_failure_logger(failure_logger)
        _logger.info("[bootstrap] JSON failure logger -> %s", log_path)

    llm_mod = importlib.import_module("T_mem.llm.llm_provider")
    emb_mod = importlib.import_module("T_mem.llm.embedding_provider")
    rr_mod = importlib.import_module("T_mem.llm.reranker_provider")

    from T_mem.llm.venus_provider import LLMProvider as _VenusLLM
    from T_mem.llm.bgem3_provider import EmbeddingProvider as _BGEM3Emb

    # Reranker selection: env T_MEM_RERANKER_MODEL controls which provider to use.
    #   unset / "trag-bge" → bge-reranker-v2-m3 via tRAG (default)
    #   "qwen3-8b" / "qwen3-6b" → Qwen3 reranker via Venus /v1/rerank
    _rr_model = os.environ.get("T_MEM_RERANKER_MODEL", "").strip()
    if _rr_model and _rr_model.startswith("qwen"):
        from T_mem.llm.qwen_reranker import RerankerProvider as _QwenReranker
        _RerankerCls = _QwenReranker
        hm_config.ExperimentConfig.reranker_config["model_name"] = _rr_model
        _logger.info("[bootstrap] reranker = QwenRerankerProvider (model=%s)", _rr_model)
    else:
        from T_mem.llm.trag_reranker import RerankerProvider as _TragReranker
        _RerankerCls = _TragReranker
        _logger.info("[bootstrap] reranker = TragRerankerProvider (bge-reranker-v2-m3)")

    llm_mod.LLMProvider = _VenusLLM
    emb_mod.EmbeddingProvider = _BGEM3Emb
    rr_mod.RerankerProvider = _RerankerCls

    for modname in list(sys.modules.keys()):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        if not modname.startswith("T_mem"):
            continue
        try:
            for attr_name, new_cls in (
                ("LLMProvider", _VenusLLM),
                ("EmbeddingProvider", _BGEM3Emb),
                ("RerankerProvider", _RerankerCls),
            ):
                if hasattr(mod, attr_name):
                    setattr(mod, attr_name, new_cls)
        except Exception as e:
            _logger.debug("[bootstrap] skip patching %s: %s", modname, e)

    _logger.info("[bootstrap] providers patched (LLM + Embedding + Reranker)")


def ensure_on_path():
    """Ensure the T_mem package is importable."""
    here = Path(__file__).resolve()
    t_mem_parent = here.parent.parent
    p = str(t_mem_parent)
    if p not in sys.path:
        sys.path.insert(0, p)
