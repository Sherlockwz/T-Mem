"""Bootstrap: sync env-driven config overrides and install the failure logger.

Providers are now first-class OpenAI-compatible implementations living in
``T_mem.llm.*`` (see ``llm_provider.py`` / ``embedding_provider.py`` /
``reranker_provider.py``); no monkey-patching is needed.

Env: T_MEM_EXPERIMENT_NAME / T_MEM_RESULTS_DIR / T_MEM_DATA_FILE /
T_MEM_NUM_CONV / T_MEM_USE_RERANKER / T_MEM_FAILURE_LOG."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger("T_mem.bootstrap")

_PATCHED = False


def patch_providers():
    """Apply env-driven config overrides and install the JSON failure logger.

    Idempotent. Kept as the single entry point all stages already call.
    """
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
        _logger.info("[bootstrap] use_reranker enabled (default)")

    _logger.info(
        "[bootstrap] experiment_name = %s",
        hm_config.ExperimentConfig.experiment_name,
    )

    log_path = os.environ.get("T_MEM_FAILURE_LOG", "").strip()
    if log_path:
        from T_mem.llm.llm_provider import JsonFailureLogger, set_global_failure_logger
        failure_logger = JsonFailureLogger(log_path)
        set_global_failure_logger(failure_logger)
        _logger.info("[bootstrap] JSON failure logger -> %s", log_path)


def ensure_on_path():
    """Ensure the T_mem package is importable."""
    here = Path(__file__).resolve()
    t_mem_parent = here.parent.parent
    p = str(t_mem_parent)
    if p not in sys.path:
        sys.path.insert(0, p)
