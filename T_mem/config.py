"""Centralized config for T_mem (paths, model ids, retrieval top-K, LLM/embedding/reranker settings)."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
_RESULTS_DIR_OVERRIDE = os.environ.get("T_MEM_RESULTS_DIR", "").strip()
RESULTS_DIR = Path(_RESULTS_DIR_OVERRIDE).resolve() if _RESULTS_DIR_OVERRIDE else (PROJECT_ROOT / "results")


MODELS: dict = {
    # NOTE: memory_build switched gpt-4.1-mini -> gpt-4o-mini for the Layer-1
    # benchmark Exp-B, so the ENTIRE pipeline (build + QA) uses one model and
    # stays cheaper. This deviates from T-mem's original config (which used
    # gpt-4.1-mini for build). Revert to "gpt-4.1-mini" to restore paper config.
    "memory_build":      "gpt-4o-mini",
    "locomo_qa":         "gpt-4o-mini",
    "locomo_judge":      "gpt-4o-mini",
    "locomo_plus_qa":    "gpt-4o",
    "locomo_plus_judge": "gemini-2.5-flash",
}

# Allow overriding the memory-construction model without editing this file, so
# the paper-faithful build (gpt-4.1-mini) can be selected via env for the
# construction-cost measurement while leaving the repo default untouched:
#   export T_MEM_MEMORY_BUILD_MODEL=gpt-4.1-mini
_mb_override = os.environ.get("T_MEM_MEMORY_BUILD_MODEL", "").strip()
if _mb_override:
    MODELS["memory_build"] = _mb_override

# Same mechanism for the QA / judge models, so the cross-judge (R1-W2) and
# stronger-backbone (R2-W1) rebuttal experiments can swap models cleanly:
#   export T_MEM_LOCOMO_QA_MODEL=gemini-2.5-pro
#   export T_MEM_LOCOMO_JUDGE_MODEL=gemini-2.5-pro
_qa_override = os.environ.get("T_MEM_LOCOMO_QA_MODEL", "").strip()
if _qa_override:
    MODELS["locomo_qa"] = _qa_override
_judge_override = os.environ.get("T_MEM_LOCOMO_JUDGE_MODEL", "").strip()
if _judge_override:
    MODELS["locomo_judge"] = _judge_override


def _detect_num_conv(dataset_path: str) -> int:
    """Read dataset_path once at import-time and return len(list).

    Locomo10 -> 10, locomo_plus stitched (401 per-sample convs) -> 401.
    Stages 2/3 iterate `range(config.num_conv)`; making this dynamic is
    what unlocks per-sample memory-library mode for LoCoMo-Plus while
    keeping LoCoMo behaviour bit-identical (locomo10.json is still 10).
    """
    import json as _json
    try:
        with open(dataset_path, "r", encoding="utf-8") as _f:
            _data = _json.load(_f)
        if isinstance(_data, list):
            return len(_data)
    except Exception:
        pass
    return 10  # safe fallback (matches locomo default)


class ExperimentConfig:
    experiment_name: str = os.environ.get("T_MEM_EXPERIMENT_NAME", "T_mem-v3")
    dataset_path: str = os.environ.get(
        "T_MEM_DATA_FILE", str(DATA_DIR / "locomo10.json")
    )
    num_conv: int = _detect_num_conv(dataset_path)

    embedding_config: dict = {
        "model_name": "bge-m3",
        "base_url": "http://v2.open.venus.oa.com/llmproxy/embeddings",
    }
    embedding_max_retries: int = 10

    retrieval_type: str = "rrf"

    # Retrieval pipeline (single source of truth).
    #
    # Two-stage design:
    #   (a) stage6 produces a *wide-recall master* sized by `scene_top_k` /
    #       `item_top_k` (default 24 / 40). The wide pool is what gets
    #       persisted to search_results.json, so a single stage6 run can be
    #       reused to feed any downstream cell whose final K is <= the wide K.
    #   (b) Right before QA, T_mem.io.truncate_search_results trims that
    #       wide pool down to `final_keep_scene` / `final_keep_item`
    #       (default 5 / 15) -- THIS is the K that actually enters the QA
    #       LLM input prompt. Paper-final M1_N15 cell == 5 scene + 15 item.
    #
    # Hyper-param sweeps over "how many scenes/items enter QA" therefore
    # only need to vary final_keep_*; the expensive stage6 step is cached.
    retrieval_config: dict = {
        "initial_candidates": int(os.environ.get("T_MEM_INITIAL_CANDIDATES", "250")),
        "topic_top_k":        int(os.environ.get("T_MEM_TOPIC_TOP_K",        "15")),
        # Wide-recall master (stage6 output size).
        "scene_top_k":        int(os.environ.get("T_MEM_SCENE_TOP_K",        "24")),
        "item_top_k":         int(os.environ.get("T_MEM_ITEM_TOP_K",         "40")),
        # Final K that actually enters the QA prompt (post-trim).
        "final_keep_scene":   int(os.environ.get("T_MEM_FINAL_KEEP_SCENE",   "5")),
        "final_keep_item":    int(os.environ.get("T_MEM_FINAL_KEEP_ITEM",    "15")),
    }

    use_reranker: bool = True
    reranker_config: dict = {
        "model_name": "bge-reranker-v2-m3",
        "base_url": "http://api.trag.woa.com",
    }
    reranker_max_retries: int = 10

    answer_type: str = "cot"
    llm_service: str = "openai"

    llm_config: dict = {
        "openai": {
            "model": MODELS["memory_build"],
            "temperature": 0.0,
            "max_tokens": 16384,
        },
    }

    max_concurrent_requests: int = 10

    @classmethod
    def experiment_dir(cls) -> Path:
        return RESULTS_DIR / cls.experiment_name

    @classmethod
    def scenes_dir(cls) -> Path:
        return cls.experiment_dir() / "scenes"

    @classmethod
    def memory_graph_dir(cls) -> Path:
        return cls.experiment_dir() / "memory_graphs"

    @classmethod
    def items_dir(cls) -> Path:
        return cls.experiment_dir() / "items"

    @classmethod
    def token_stats_dir(cls) -> Path:
        return cls.experiment_dir() / "token_stats"

    @classmethod
    def bm25_index_dir(cls) -> Path:
        return cls.experiment_dir() / "bm25_index"

    @classmethod
    def vectors_dir(cls) -> Path:
        return cls.experiment_dir() / "vectors"
