"""Hierarchical retrieval: topic → scene → item, with BM25 / vector / RRF fusion and optional reranking."""

import sys
import os
import pickle
import json
import asyncio
import threading
from pathlib import Path
import nltk
import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from T_mem.bootstrap import patch_providers  # noqa: E402
patch_providers()

from T_mem.llm.embedding_provider import EmbeddingProvider
from T_mem.llm.reranker_provider import RerankerProvider
from T_mem.config import ExperimentConfig
from T_mem.retrievers.trigger_recaller import TriggerRecaller

console = Console()

# RRF fusion constant (k in the RRF formula; standard choice is 60).
RRF_K = 60


# L1 Trigger recall hook (env-gated, default ON).
# Env: T_MEM_L1_TRIGGER_ENABLED ('0'/'false'/'no'/'off' to DISABLE; else ON).
# T_MEM_L1_TRIGGER_DIR (default <experiment_dir>/trigger/), TOPK (10), GATE (0.85).
# When enabled, L1 recall item_ids are union'd into Layer-3 connected_items BEFORE
# BM25/Vector/RRF scoring; missing artefacts -> silent baseline fallback with warning.
_L1_RECALLER_CACHE: Dict[int, Optional[TriggerRecaller]] = {}
_L1_RECALLER_CACHE_LOCK = threading.Lock()


def _is_l1_trigger_enabled() -> bool:
    """Default ON; only explicit opt-out disables the feature."""
    raw = os.environ.get("T_MEM_L1_TRIGGER_ENABLED", "").strip().lower()
    # explicit off
    if raw in ("0", "false", "no", "off"):
        return False
    # everything else (including unset) -> on
    return True


def _l1_trigger_topk() -> int:
    raw = os.environ.get("T_MEM_L1_TRIGGER_TOPK", "").strip()
    if not raw:
        return 10
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _l1_trigger_gate() -> float:
    raw = os.environ.get("T_MEM_L1_TRIGGER_GATE", "").strip()
    if not raw:
        return 0.85
    try:
        v = float(raw)
        return max(0.0, min(1.0, v))
    except ValueError:
        return 0.85


def _load_l1_trigger_recaller(
    conv_id: int,
    memory_graph_dir: Path,
) -> Optional[TriggerRecaller]:
    """Lazy-load the per-conv L1 TriggerRecaller.

    Trigger artefacts location priority:
      1. T_MEM_L1_TRIGGER_DIR env (absolute or relative)
      2. <memory_graph_dir>.parent / "trigger"  (i.e. the default
         output path of build_trigger.py, which sits next to the
         memory-graph dir under <experiment_dir>/)

    Returns None when the feature is disabled, the directory is missing,
    or the per-conv artefacts for this conv aren't built yet. Thread-safe
    via a per-process lock; populated entries are cached so recall does
    not re-read the NPZ each question.
    """
    if not _is_l1_trigger_enabled():
        return None

    # Fast path — cached (including cached None for missing artefacts).
    if conv_id in _L1_RECALLER_CACHE:
        return _L1_RECALLER_CACHE[conv_id]

    with _L1_RECALLER_CACHE_LOCK:
        if conv_id in _L1_RECALLER_CACHE:
            return _L1_RECALLER_CACHE[conv_id]

        override_dir = os.environ.get("T_MEM_L1_TRIGGER_DIR", "").strip()
        if override_dir:
            trigger_dir = Path(override_dir)
        else:
            trigger_dir = memory_graph_dir.parent / "trigger"

        if not trigger_dir.exists():
            print(
                f"[L1_TRIGGER] WARNING: trigger_dir not found: {trigger_dir} "
                f"— disabling L1 recall for conv {conv_id}"
            )
            _L1_RECALLER_CACHE[conv_id] = None
            return None

        try:
            recaller = TriggerRecaller.from_dir(trigger_dir, conv_id)
        except Exception as e:  # noqa: BLE001
            print(
                f"[L1_TRIGGER] WARNING: failed to load conv {conv_id} from "
                f"{trigger_dir}: {e} — disabling L1 recall for conv {conv_id}"
            )
            _L1_RECALLER_CACHE[conv_id] = None
            return None

        if recaller is None:
            print(
                f"[L1_TRIGGER] conv {conv_id}: no trigger artefacts in "
                f"{trigger_dir} — L1 recall disabled for this conv"
            )
        else:
            print(
                f"[L1_TRIGGER] conv {conv_id}: loaded {len(recaller.trigger_ids)} "
                f"L1 triggers from {trigger_dir} "
                f"(topk={_l1_trigger_topk()}, gate={_l1_trigger_gate():.2f})"
            )
        _L1_RECALLER_CACHE[conv_id] = recaller
        return recaller


# L2L3 associative-recall hook (env-gated, default OFF).
# When L2L3_ASSOC_TOPK_JSON points to a per-qa top-K file, Layer 2 unions the
# top-K scene_ids into connected_scenes BEFORE BM25/Vector/RRF; rest unchanged.
_L2L3_ASSOC_CACHE: Optional[Dict[Tuple[int, str], frozenset]] = None
_L2L3_ASSOC_CACHE_LOADED: bool = False
_L2L3_ASSOC_LOCK = threading.Lock()


def _load_l2l3_assoc_topk() -> Optional[Dict[Tuple[int, str], frozenset]]:
    """Lazy-load the per-qa top-K association map from env
    L2L3_ASSOC_TOPK_JSON. Returns None if env is unset or file missing.

    Schema expected (see build_l2l3_topk_per_qa.py):
      {
        "meta": {...},
        "per_qa": [
          {"conv_id": int, "question": str, "topk": [scene_id, ...]},
          ...
        ]
      }

    Thread-safety: stage6 calls this from multiple worker threads via
    asyncio.run_in_executor(None, ...). A naive double-checked-locking
    pattern (set LOADED=True first, then assign CACHE) has a race window
    where other threads observe LOADED=True while CACHE is still None,
    silently disabling the hook for all concurrent conversations. We use
    a Lock plus the invariant "CACHE is fully assigned before LOADED
    flips to True" to eliminate the race.
    """
    global _L2L3_ASSOC_CACHE, _L2L3_ASSOC_CACHE_LOADED
    # Fast path — once loading is truly complete, no lock needed.
    if _L2L3_ASSOC_CACHE_LOADED:
        return _L2L3_ASSOC_CACHE
    with _L2L3_ASSOC_LOCK:
        # Re-check under lock (another thread may have finished).
        if _L2L3_ASSOC_CACHE_LOADED:
            return _L2L3_ASSOC_CACHE
        path = os.environ.get("L2L3_ASSOC_TOPK_JSON", "").strip()
        if not path:
            _L2L3_ASSOC_CACHE = None
            _L2L3_ASSOC_CACHE_LOADED = True
            return None
        p = Path(path)
        if not p.exists():
            print(f"[L2L3_ASSOC] WARNING: L2L3_ASSOC_TOPK_JSON points to missing file: {p} — disabling")
            _L2L3_ASSOC_CACHE = None
            _L2L3_ASSOC_CACHE_LOADED = True
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[L2L3_ASSOC] WARNING: failed to load {p}: {e} — disabling")
            _L2L3_ASSOC_CACHE = None
            _L2L3_ASSOC_CACHE_LOADED = True
            return None
        out: Dict[Tuple[int, str], frozenset] = {}
        for rec in payload.get("per_qa", []) or []:
            cid = rec.get("conv_id")
            q = rec.get("question")
            topk = rec.get("topk") or []
            if cid is None or not q:
                continue
            out[(int(cid), q)] = frozenset(x for x in topk if x)
        meta = payload.get("meta", {}) or {}
        print(f"[L2L3_ASSOC] loaded {len(out)} per-qa entries from {p} "
              f"(topk={meta.get('topk')}, rrf_k={meta.get('rrf_k')})")
        # Critical: assign CACHE first, then set LOADED=True. Other
        # threads on the fast path must never observe LOADED=True with
        # CACHE still None.
        _L2L3_ASSOC_CACHE = out
        _L2L3_ASSOC_CACHE_LOADED = True
        return out


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Dict, float]]],
    top_n: int,
    k: int = RRF_K
) -> List[Tuple[Dict, float]]:
    """Reciprocal Rank Fusion: RRF_score(d) = Σ 1 / (k + rank(d))."""
    doc_scores = {}  # doc_id -> {"doc": doc, "score": rrf_score}
    
    for results in results_list:
        for rank, (doc, _) in enumerate(results):
            doc_id = doc.get("id")
            if doc_id is None:
                # Fall back to a hash of the doc payload when no id is present.
                doc_id = hash(str(doc))
            
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += rrf_score
            else:
                doc_scores[doc_id] = {"doc": doc, "score": rrf_score}
    
    sorted_results = sorted(
        [(item["doc"], item["score"]) for item in doc_scores.values()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results[:top_n]


# Retrieval prompt template (fixed output: scene + item only).
# topic layer is never emitted to the QA LLM.
RETRIEVAL_TEMPLATE = (
    "Scene memories for conversation between {speaker_1} and {speaker_2}:"
    "\n## Relevant Scenes:\n{scenes}"
    "\n## Relevant Items:\n{items}"
)


def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK tokenization, consistent with index building

    Args:
        text: Text to tokenize
        stemmer: Stemmer
        stop_words: Set of stop words

    Returns:
        List of processed tokens
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())
    
    processed_tokens = [
        stemmer.stem(token) 
        for token in tokens 
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]
    
    return processed_tokens


def search_with_bm25(
    query: str, 
    bm25, 
    docs: List[Dict], 
    doc_type: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """BM25 retrieval filtered by doc_type; returns [(doc.data, score)] sorted by score."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)
    
    if not tokenized_query:
        print(f"Warning: Query is empty after tokenization for {doc_type}")
        return []

    # Get scores for all documents
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Filter documents of the specified type and extract the data field
    filtered_results = [
        (doc.get("data", doc), score) for doc, score in zip(docs, doc_scores)
        if doc.get("type") == doc_type
    ]
    
    # Sort by score
    sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    return sorted_results[:top_n]


def search_with_emb(
    query: str,
    emb_index: List[Dict],
    embedding_provider: EmbeddingProvider,
    doc_type: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """Vector retrieval filtered by doc_type; returns [(doc.data, cosine)] sorted."""
    query_vec = np.array(embedding_provider.embed([query])[0])
    
    # Filter documents of the specified type
    filtered_items = [
        item for item in emb_index
        if item.get("type") == doc_type
    ]
    
    if not filtered_items:
        return []
    
    # Extract vectors and corresponding data
    embeddings = [item["embedding"] for item in filtered_items]
    embeddings_np = np.array(embeddings)
    
    # L2 norm normalization
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

    # Calculate cosine similarity
    scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)
    
    # Build result list (return complete data dict)
    results_with_scores = [
        (filtered_items[i]["data"], scores[i])
        for i in range(len(filtered_items))
    ]
    
    # Sort by score
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_n]


def rerank_results(
    query: str,
    results: List[Tuple[Dict, float]],
    reranker_provider: RerankerProvider,
    text_field: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """Rerank retrieval results via a cross-encoder reranker."""
    if not results:
        return []
    
    # Extract documents and text
    docs = []
    doc_texts = []
    for doc, score in results:
        text = doc.get(text_field, "")
        if text:
            docs.append(doc)
            doc_texts.append(text)
    
    if not doc_texts:
        return []
    
    # Prepare query list
    queries = [query] * len(doc_texts)
    
    # Get reranking scores
    rerank_scores = reranker_provider.rerank(queries, doc_texts)
    
    # Build new result list
    reranked_results = list(zip(docs, rerank_scores))
    
    # Sort by reranking score
    sorted_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_n]


def get_connected_scenes(
    topic_ids: Set[str],
    memory_graph: Dict[str, Any]
) -> Set[str]:
    """Expand Layer-1 topics → set of scene ids via `TopicNode.scene_ids`."""
    connected: Set[str] = set()
    topics_dict = memory_graph.get("topics", {})
    for topic_id in topic_ids:
        topic = topics_dict.get(topic_id, {})
        if not topic:
            continue
        for sc_id in topic.get("scene_ids", []) or []:
            if sc_id:
                connected.add(sc_id)
    return connected


def get_connected_items(
    scene_ids: Set[str],
    memory_graph: Dict[str, Any]
) -> Set[str]:
    """Expand Layer-2 scenes → set of item ids via `SceneNode.item_ids`."""
    connected: Set[str] = set()
    scenes_dict = memory_graph.get("scenes", {})
    for scene_id in scene_ids:
        scene = scenes_dict.get(scene_id, {})
        if not scene:
            continue
        for it_id in scene.get("item_ids", []) or []:
            if it_id:
                connected.add(it_id)
    return connected


def hierarchical_retrieval(
    query: str,
    memory_graph: Dict[str, Any],
    bm25=None,
    docs=None,
    emb_index=None,
    embedding_provider: EmbeddingProvider = None,
    reranker_provider: RerankerProvider = None,
    config: ExperimentConfig = None,
    l2l3_assoc_ids: Optional[Set[str]] = None,
    l1_trigger_recaller: Optional[TriggerRecaller] = None,
) -> Dict[str, List[Tuple[Dict, float]]]:
    """Top-down three-layer retrieval: Topic → Scene → Item.

    When `l2l3_assoc_ids` is provided it is union'd into Layer-2's scene pool
    before BM25/Vector/RRF scoring (L2L3 associative-recall hook).

    When `l1_trigger_recaller` is provided it is invoked at Layer 3 to
    surface extra item_ids via concept/bridge/joint L1 trigger cosine
    match; surviving item_ids are union'd into `connected_items` BEFORE
    BM25/Vector/RRF scoring, leaving downstream rerank + top-K unchanged.
    Requires a non-None `embedding_provider` to embed the query.
    """
    results = {
        "topics": [],
        "scenes": [],
        "items": []
    }

    # Retrieval log: captures the full retrieval flow for debugging
    retrieval_log = {
        "query": query,
        "config": {},
        "layer1_topic": {},
        "layer2_scene": {},
        "layer3_item": {},
    }

    # Post-dismantle: topic is never emitted to the QA LLM. We always run all
    # three layers (topic -> scene -> item) because topic is still the
    # Layer 1 filter; only the final context formatting skips topic.

    # Unified retrieval top-K for ALL question types (no more keyword-based
    # detect_question_type dispatch). See module-level NOTE at top of file.
    topic_top_k = config.retrieval_config["topic_top_k"]
    scene_top_k = config.retrieval_config["scene_top_k"]
    item_top_k = config.retrieval_config["item_top_k"]
    initial_candidates = config.retrieval_config["initial_candidates"]
    # === Layer 1: Retrieve relevant topics ===

    print(f"  [Layer 1] Retrieving relevant topics...")
    retrieve_top_n = initial_candidates if config.use_reranker else topic_top_k
    
    # Select retrieval method based on retrieval type: keyword (BM25 only), vector (vector only), rrf (fusion)
    retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
    use_emb = retrieval_type in ('vector', 'rrf')
    use_bm25 = retrieval_type in ('keyword', 'rrf')
    use_rrf = retrieval_type == 'rrf'

    retrieval_log["config"] = {
        "topic_top_k": topic_top_k,
        "scene_top_k": scene_top_k,
        "item_top_k": item_top_k,
        "initial_candidates": initial_candidates,
        "use_reranker": config.use_reranker,
        "retrieval_type": retrieval_type,
    }

    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval: use both BM25 and vector retrieval, then fuse
        bm25_topic_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        emb_topic_results = search_with_emb(
            query=query,
            emb_index=emb_index,
            embedding_provider=embedding_provider,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        topic_results = reciprocal_rank_fusion(
            [bm25_topic_results, emb_topic_results],
            top_n=retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_topic_results)}) + Vector({len(emb_topic_results)}) → {len(topic_results)}")
        retrieval_log["layer1_topic"]["bm25"] = [(d.get("id",""), float(round(s,3))) for d,s in bm25_topic_results]
        retrieval_log["layer1_topic"]["emb"] = [(d.get("id",""), float(round(s,3))) for d,s in emb_topic_results]
        retrieval_log["layer1_topic"]["rrf"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    elif use_emb and emb_index:
        topic_results = search_with_emb(
            query=query,
            emb_index=emb_index,
            embedding_provider=embedding_provider,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        print(f"    [Vector] Retrieved {len(topic_results)} topics")
        retrieval_log["layer1_topic"]["emb"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    else:
        topic_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        print(f"    [BM25] Retrieved {len(topic_results)} topics")
        retrieval_log["layer1_topic"]["bm25"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]

    # Rerank topics
    pre_rerank_topics = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    if config.use_reranker and topic_results and reranker_provider:
        try:
            topic_results = rerank_results(
                query=query,
                results=topic_results,
                reranker_provider=reranker_provider,
                text_field="summary",
                top_n=topic_top_k
            )
            retrieval_log["layer1_topic"]["reranked"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
        except Exception as e:
            print(f"    [WARNING] Reranker failed, using original results: {e}")
            topic_results = topic_results[:topic_top_k]
    else:
        topic_results = topic_results[:topic_top_k]
    retrieval_log["layer1_topic"]["pre_rerank"] = pre_rerank_topics
    retrieval_log["layer1_topic"]["final"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    retrieval_log["layer1_topic"]["final_titles"] = [d.get("title","")[:60] for d,s in topic_results]

    results["topics"] = topic_results
    relevant_topic_ids = {topic_data["id"] for topic_data, _ in topic_results}
    print(f"    Found {len(relevant_topic_ids)} relevant topics")

    if not relevant_topic_ids:
        print("    Warning: No relevant topics found, skipping subsequent retrieval")
        results["retrieval_log"] = retrieval_log
        return results

    # --- Ablation A5/A6: optionally override relevant_topic_ids to the full
    # set of topic ids, so Layer 2 receives the entire scene pool via
    # get_connected_scenes. Applies when either
    #   T_MEM_FLAT_FACT_RETRIEVAL=1   (A5: flat item retrieval)
    # or
    #   T_MEM_SKIP_TOPIC_LAYER=1      (A6: skip topic layer)
    # is set. For A5 we ALSO override connected_items later at Layer 3.
    _ablate_flat_fact = os.environ.get("T_MEM_FLAT_FACT_RETRIEVAL", "").strip().lower() in ("1", "true", "yes", "on")
    _ablate_skip_topic = os.environ.get("T_MEM_SKIP_TOPIC_LAYER", "").strip().lower() in ("1", "true", "yes", "on")
    if _ablate_flat_fact or _ablate_skip_topic:
        _all_topic_ids = set((memory_graph.get("topics", {}) or {}).keys())
        _n_orig_topics = len(relevant_topic_ids)
        relevant_topic_ids = _all_topic_ids
        retrieval_log["ablation"] = {
            "flat_fact_retrieval": _ablate_flat_fact,
            "skip_topic_layer": _ablate_skip_topic,
            "layer1_override": {
                "orig_relevant_topic_ids": _n_orig_topics,
                "expanded_to_all": len(relevant_topic_ids),
            },
        }
        print(f"    [ABLATION] layer1 topic override: {_n_orig_topics} -> {len(relevant_topic_ids)} (all)")

    # === Layer 2: Get connected scenes from relevant topics ===
    # (Previous code here had a short-circuit on
    #  `need_scene_output` / `need_item_output` / `output_type`, but those
    #  variables were never defined anywhere in this function — dead / broken
    #  code left over from an abandoned optimization. Removed so Layer 2+3
    #  always run, which is the intended behavior post-dismantle.)

    print(f"  [Layer 2] Retrieving scenes from relevant topics...")
    connected_scenes = get_connected_scenes(relevant_topic_ids, memory_graph)
    print(f"    Found {len(connected_scenes)} scenes via graph-edge connections")
    retrieval_log["layer2_scene"]["connected_count"] = len(connected_scenes)

    # === L2L3 associative-recall union (env-gated, no-op if off) ===
    # Union the per-question L2L3 top-K scene_ids with the topic->scene
    # expansion pool; the union is then fed to downstream BM25+Vector RRF
    # plus rerank. We keep only ids belonging to the current conv
    # memory graph (normally identical; the scenes_dict filter is purely
    # defensive against dirty ids in external json).
    if l2l3_assoc_ids:
        _mg_scenes = set((memory_graph.get("scenes", {}) or {}).keys())
        _valid_assoc = {e for e in l2l3_assoc_ids if e in _mg_scenes}
        # Snapshot the pre-union topic pool so downstream analysis can
        # decide, per-id, whether a given L2L3 scene was an overlap with
        # the topic pool or a purely-new contribution. Without this dump
        # the per-id split is not recoverable from the final retrieval log
        # (pre_rerank is the *union*, which conflates both buckets).
        _topic_pool_before = set(connected_scenes)
        _before = len(_topic_pool_before)
        _overlap_ids = _valid_assoc & _topic_pool_before
        _added_ids = _valid_assoc - _topic_pool_before
        if isinstance(connected_scenes, set):
            connected_scenes = connected_scenes | _valid_assoc
        else:
            connected_scenes = set(connected_scenes) | _valid_assoc
        _added = len(_added_ids)
        _overlap = len(_overlap_ids)
        retrieval_log["layer2_scene"]["l2l3_assoc"] = {
            "n_assoc_input": len(l2l3_assoc_ids),
            "n_assoc_valid": len(_valid_assoc),
            "n_overlap_with_topic_pool": _overlap,
            "n_added": _added,
            "connected_after_union": len(connected_scenes),
            # Per-id lists (25-char truncation is applied by the same
            # downstream log-serialization as every other scene id in
            # the retrieval_log; we keep full ids here — the consumer is
            # free to truncate to match pre_rerank/reranked).
            "overlap_ids": sorted(_overlap_ids),
            "added_ids": sorted(_added_ids),
        }
        print(
            f"    [L2L3_ASSOC] union: topic_pool={_before} + l2l3={len(l2l3_assoc_ids)}"
            f" (valid={len(_valid_assoc)}, overlap={_overlap}) → {_before + _added}"
        )

    if not connected_scenes:
        print("    Warning: No connected scenes found, skipping item retrieval")
        results["retrieval_log"] = retrieval_log
        return results

    # Retrieve within connected scenes
    # scene_top_k was set at the beginning of the function based on question type
    scene_retrieve_top_n = initial_candidates if config.use_reranker else scene_top_k

    # Define helper function for BM25 scene retrieval
    def search_scenes_bm25():
        bm25_scene_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="scene",
            top_n=scene_retrieve_top_n * 2  # Retrieve more, will be filtered later
        )
        # Filter connected scenes (using BM25 scores only, without weighting)
        filtered_results = []
        for doc, score in bm25_scene_results:
            doc_data = doc.get("data", doc) if isinstance(doc, dict) and "data" in doc else doc
            scene_id = doc_data.get("id")
            if scene_id in connected_scenes:
                # Use BM25 score directly
                filtered_results.append((doc_data, score))
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)[:scene_retrieve_top_n]

    # Define helper function for vector scene retrieval
    def search_scenes_emb():
        filtered_emb_index = [
            item for item in emb_index
            if item.get("type") == "scene" and item.get("id") in connected_scenes
        ]
        if not filtered_emb_index:
            return []
        
        query_vec = np.array(embedding_provider.embed([query])[0])
        embeddings = [item["embedding"] for item in filtered_emb_index]
        embeddings_np = np.array(embeddings)
        
        # L2 norm normalization
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

        scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)

        # Use vector similarity scores directly
        emb_results = []
        for i, item in enumerate(filtered_emb_index):
            emb_results.append((item["data"], scores[i]))

        return sorted(emb_results, key=lambda x: x[1], reverse=True)[:scene_retrieve_top_n]

    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval
        bm25_scene_results = search_scenes_bm25()
        emb_scene_results = search_scenes_emb()
        scene_results = reciprocal_rank_fusion(
            [bm25_scene_results, emb_scene_results],
            top_n=scene_retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_scene_results)}) + Vector({len(emb_scene_results)}) → {len(scene_results)}")
        retrieval_log["layer2_scene"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in bm25_scene_results[:20]]
        retrieval_log["layer2_scene"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in emb_scene_results[:20]]
        retrieval_log["layer2_scene"]["rrf"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results[:20]]
    elif use_emb and emb_index:
        scene_results = search_scenes_emb()
        print(f"    [Vector] Retrieved {len(scene_results)} scenes")
        retrieval_log["layer2_scene"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results[:20]]
    else:
        scene_results = search_scenes_bm25()
        print(f"    [BM25] Retrieved {len(scene_results)} scenes")
        retrieval_log["layer2_scene"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results[:20]]

    # Rerank scenes
    pre_rerank_scenes = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results]
    if config.use_reranker and scene_results and reranker_provider:
        try:
            scene_results = rerank_results(
                query=query,
                results=scene_results,
                reranker_provider=reranker_provider,
                text_field="scene_description",
                top_n=scene_top_k
            )
            retrieval_log["layer2_scene"]["reranked"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results]
        except Exception as e:
            print(f"    [WARNING] Scene reranker failed, using original results: {e}")
            scene_results = scene_results[:scene_top_k]
    else:
        scene_results = scene_results[:scene_top_k]
    retrieval_log["layer2_scene"]["pre_rerank"] = pre_rerank_scenes
    retrieval_log["layer2_scene"]["final"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in scene_results]
    retrieval_log["layer2_scene"]["final_subjects"] = [d.get("subject","")[:50] for d,s in scene_results]

    results["scenes"] = scene_results
    relevant_scene_ids = {scene_data["id"] for scene_data, _ in scene_results}
    print(f"    Retrieved {len(relevant_scene_ids)} relevant scenes")

    if not relevant_scene_ids:
        print("    Warning: No relevant scenes found, skipping item retrieval")
        results["retrieval_log"] = retrieval_log
        return results

    # === Layer 3: Get connected items from relevant scenes ===

    print(f"  [Layer 3] Retrieving items from relevant scenes...")
    connected_items = get_connected_items(relevant_scene_ids, memory_graph)
    # A5 ablation: expand connected_items to the full item set of this
    # sample, so Layer 3 performs a flat retrieval over all items.
    if _ablate_flat_fact:
        _n_before = len(connected_items)
        connected_items = set((memory_graph.get("items", {}) or {}).keys())
        _abl = retrieval_log.get("ablation") or {}
        _abl["layer3_override"] = {
            "orig_connected_items": _n_before,
            "expanded_to_all": len(connected_items),
        }
        retrieval_log["ablation"] = _abl
        print(f"    [ABLATION] layer3 item override: {_n_before} -> {len(connected_items)} (all)")
    print(f"    Found {len(connected_items)} items via graph-edge connections")
    retrieval_log["layer3_item"]["connected_count"] = len(connected_items)

    # -------- L1 trigger associative recall (Layer-3 union, env-gated) --------
    # If an L1 recaller is attached AND we have an embedding_provider, embed
    # the query (BGE-M3 space), recall top-K L1 triggers through tri-view
    # nanmax with a HARD cosine gate (default 0.85), and union the resulting
    # item_ids into connected_items. Downstream BM25/Vector/RRF + rerank +
    # item_top_k still decide the final 25.
    if l1_trigger_recaller is not None and embedding_provider is not None:
        try:
            _topk = _l1_trigger_topk()
            _gate = _l1_trigger_gate()
            _q_vec = np.asarray(
                embedding_provider.embed([query])[0], dtype=np.float32
            )
            _l1_result = l1_trigger_recaller.recall(
                _q_vec,
                max_top_triggers=_topk,
                min_cosine_gate=_gate,
            )
            # Only union item_ids that actually exist in the item store.
            _mg_items = set((memory_graph.get("items", {}) or {}).keys())
            _l1_raw_ids = set(_l1_result.item_ids)
            _l1_valid_ids = _l1_raw_ids & _mg_items
            _l1_new_ids = _l1_valid_ids - connected_items
            _n_before = len(connected_items)
            if _l1_new_ids:
                connected_items = connected_items | _l1_new_ids
            retrieval_log["layer3_item"]["l1_trigger"] = {
                "enabled": True,
                "triggered": bool(_l1_result.triggered),
                "reason": _l1_result.reason,
                "topk": _topk,
                "gate": _gate,
                "top1_cosine": float(_l1_result.top1_cosine),
                "n_triggers_fired": len(_l1_result.triggers),
                "n_items_recalled": len(_l1_raw_ids),
                "n_items_valid": len(_l1_valid_ids),
                "n_items_new": len(_l1_new_ids),
                "connected_before": _n_before,
                "connected_after": len(connected_items),
            }
            print(
                f"    [L1_TRIGGER] fired={_l1_result.triggered} "
                f"cos1={_l1_result.top1_cosine:.3f} "
                f"triggers={len(_l1_result.triggers)} "
                f"items={len(_l1_raw_ids)} (new={len(_l1_new_ids)}) "
                f"connected: {_n_before} -> {len(connected_items)}"
            )
        except Exception as _l1_e:  # noqa: BLE001
            retrieval_log["layer3_item"]["l1_trigger"] = {
                "enabled": True,
                "error": f"{type(_l1_e).__name__}: {_l1_e}",
            }
            print(f"    [L1_TRIGGER] WARNING: recall failed: {_l1_e}")

    if not connected_items:
        print("    Warning: No connected items found")
        results["retrieval_log"] = retrieval_log
        return results

    # item_top_k was set at the beginning of the function based on question type
    item_retrieve_top_n = initial_candidates if config.use_reranker else item_top_k
    
    # Define helper function for BM25 item retrieval
    def search_items_bm25():
        bm25_item_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="item",
            top_n=item_retrieve_top_n * 2
        )
        # Filter connected items (using BM25 scores only, without weighting)
        filtered_results = []
        for doc, score in bm25_item_results:
            doc_data = doc.get("data", doc) if isinstance(doc, dict) and "data" in doc else doc
            item_id = doc_data.get("id")
            if item_id in connected_items:
                # Use BM25 score directly
                filtered_results.append((doc_data, score))
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)[:item_retrieve_top_n]

    # Define helper function for vector item retrieval
    def search_items_emb():
        filtered_emb_index = [
            item for item in emb_index
            if item.get("type") == "item" and item.get("id") in connected_items
        ]
        if not filtered_emb_index:
            return []

        query_vec = np.array(embedding_provider.embed([query])[0])
        embeddings = [item["embedding"] for item in filtered_emb_index]
        embeddings_np = np.array(embeddings)

        # L2 norm normalization
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

        scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)

        # Use vector similarity scores directly
        emb_results = []
        for i, item in enumerate(filtered_emb_index):
            emb_results.append((item["data"], scores[i]))

        return sorted(emb_results, key=lambda x: x[1], reverse=True)[:item_retrieve_top_n]

    # Retrieve within connected items
    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval
        bm25_item_results = search_items_bm25()
        emb_item_results = search_items_emb()
        item_results = reciprocal_rank_fusion(
            [bm25_item_results, emb_item_results],
            top_n=item_retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_item_results)}) + Vector({len(emb_item_results)}) → {len(item_results)}")
        retrieval_log["layer3_item"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in bm25_item_results[:20]]
        retrieval_log["layer3_item"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in emb_item_results[:20]]
        retrieval_log["layer3_item"]["rrf"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results[:20]]
    elif use_emb and emb_index:
        item_results = search_items_emb()
        print(f"    [Vector] Retrieved {len(item_results)} items")
        retrieval_log["layer3_item"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results[:20]]
    else:
        item_results = search_items_bm25()
        print(f"    [BM25] Retrieved {len(item_results)} items")
        retrieval_log["layer3_item"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results[:20]]

    # Rerank items
    pre_rerank_items = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results]
    if config.use_reranker and item_results and reranker_provider:
        try:
            item_texts = []
            item_docs = []
            for doc, score in item_results:
                text_parts = []
                if doc.get("content"):
                    text_parts.append(doc["content"])
                kw = doc.get("keywords") or []
                if isinstance(kw, list) and kw:
                    text_parts.append(" ".join(kw))

                item_text = " ".join(text_parts)
                if item_text:
                    item_docs.append(doc)
                    item_texts.append(item_text)

            if item_texts:
                queries = [query] * len(item_texts)
                rerank_scores = reranker_provider.rerank(queries, item_texts)
                item_results = sorted(
                    zip(item_docs, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:item_top_k]
                retrieval_log["layer3_item"]["reranked"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results]
        except Exception as e:
            print(f"    [WARNING] Item reranker failed, using original results: {e}")
            item_results = item_results[:item_top_k]
    else:
        item_results = item_results[:item_top_k]
    retrieval_log["layer3_item"]["pre_rerank"] = pre_rerank_items
    retrieval_log["layer3_item"]["final"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in item_results]
    retrieval_log["layer3_item"]["final_contents"] = [d.get("content","")[:60] for d,s in item_results]

    results["items"] = item_results
    results["retrieval_log"] = retrieval_log
    print(f"    Retrieved {len(item_results)} relevant items")

    return results


def format_hierarchical_results(
    results: Dict[str, List[Tuple[Dict, float]]],
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Format retrieval results as QA context.

    Topic layer is never emitted; only scene + item are rendered.
    """
    # Format scenes
    scene_texts = []
    for idx, (scene_data, score) in enumerate(results.get("scenes", []), 1):
        scene_desc = scene_data.get('scene_description', '')
        timestamp = scene_data.get('timestamp', '')

        scene_text = f"[Scene {idx}] {scene_desc}"
        if timestamp:
            scene_text += f"\n  Time: {timestamp}"

        scene_texts.append(scene_text)

    scenes_str = "\n\n".join(scene_texts) if scene_texts else "No relevant scenes found."

    # Format items
    item_texts = []
    for idx, (item_data, score) in enumerate(results.get("items", []), 1):
        content = item_data.get('content', '')
        item_text = f"[Item {idx}] {content}"
        temporal = item_data.get('temporal', '')
        spatial = item_data.get('spatial', '')
        if temporal:
            item_text += f"\n  Time: {temporal}"
        if spatial:
            item_text += f"\n  Location: {spatial}"

        item_texts.append(item_text)

    items_str = "\n\n".join(item_texts) if item_texts else "No relevant items found."

    context = RETRIEVAL_TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_2=speaker_b,
        scenes=scenes_str,
        items=items_str,
    )

    return context


def get_query_count(conversation_data: Dict[str, Any]) -> int:
    """Count queries to process in a conversation (excluding category 5 adversarial)."""
    if "qa" not in conversation_data:
        return 0
    
    count = 0
    for qa_pair in conversation_data["qa"]:
        if qa_pair.get("question") and qa_pair.get("category") != 5:
            count += 1
    return count


def _extract_retrieved_dia_ids(
    hierarchical_results: Dict[str, Any],
    memory_graph: Dict[str, Any],
) -> Dict[str, list]:
    """[BENCHMARK ADD-ON] Trace retrieved scenes/items back to original turn ids.

    Non-invasive: reads the in-memory retrieval result + memory_graph and returns
    the original dia_ids (== our benchmark turn_ids) of everything retrieved.
    Used only to enrich the output record; does NOT change any retrieval logic.

    Returns {retrieved_turn_ids, retrieved_scene_ids, retrieved_item_ids,
             turn_ids_from_scenes, turn_ids_from_items}.
    """
    scenes_g = memory_graph.get("scenes", {}) or {}
    items_g = memory_graph.get("items", {}) or {}

    hit_scene_ids: list[str] = []
    for scene_data, _score in hierarchical_results.get("scenes", []) or []:
        sid = scene_data.get("id") if isinstance(scene_data, dict) else None
        if sid:
            hit_scene_ids.append(sid)

    hit_item_ids: list[str] = []
    item_scene_ids: list[str] = []
    for item_data, _score in hierarchical_results.get("items", []) or []:
        iid = item_data.get("id") if isinstance(item_data, dict) else None
        if iid:
            hit_item_ids.append(iid)
            node = items_g.get(iid, {})
            for s in (node.get("scene_ids") or []):
                item_scene_ids.append(s)

    def _dia_ids_of_scene(sid: str) -> list[str]:
        node = scenes_g.get(sid, {})
        out = []
        for od in (node.get("original_data") or []):
            d = od.get("dia_id") if isinstance(od, dict) else None
            if d:
                out.append(d)
        return out

    turns_from_scenes: list[str] = []
    for sid in hit_scene_ids:
        turns_from_scenes.extend(_dia_ids_of_scene(sid))

    turns_from_items: list[str] = []
    for sid in item_scene_ids:
        turns_from_items.extend(_dia_ids_of_scene(sid))

    # union, order-preserving
    seen = set()
    all_turns = []
    for t in turns_from_scenes + turns_from_items:
        if t not in seen:
            seen.add(t)
            all_turns.append(t)

    return {
        "retrieved_turn_ids": all_turns,
        "retrieved_scene_ids": list(dict.fromkeys(hit_scene_ids + item_scene_ids)),
        "retrieved_item_ids": hit_item_ids,
        "turn_ids_from_scenes": list(dict.fromkeys(turns_from_scenes)),
        "turn_ids_from_items": list(dict.fromkeys(turns_from_items)),
    }


def process_single_conversation_retrieval(
    conv_id: int,
    conversation_data: Dict[str, Any],
    config: ExperimentConfig,
    memory_graph_dir: Path,
    index_dir: Path,
    embedding_provider: Optional[EmbeddingProvider],
    reranker_provider: Optional[RerankerProvider],
    progress_callback: Optional[callable] = None
) -> tuple[str, List[Dict[str, Any]]]:
    """Run retrieval for all QA pairs of a single conversation."""
    try:
        conv_id_str = f"locomo_exp_user_{conv_id}"
        speaker_a = conversation_data["conversation"].get("speaker_a", "Speaker A")
        speaker_b = conversation_data["conversation"].get("speaker_b", "Speaker B")
        
        if "qa" not in conversation_data:
            console.print(f"  [yellow][!] Conversation {conv_id}: 'qa' field not found[/yellow]")
            return (conv_id_str, [])
        
        # === Load memory-graph data ===
        memory_graph_file = memory_graph_dir / f"memory_graph_conv_{conv_id}.json"
        if not memory_graph_file.exists():
            console.print(f"  [yellow][!] Conversation {conv_id}: memory-graph file not found[/yellow]")
            return (conv_id_str, [])

        with open(memory_graph_file, "r", encoding="utf-8") as f:
            memory_graph = json.load(f)
        
        # === Load indexes (hybrid retrieval requires loading both BM25 and vector indexes) ===
        bm25 = None
        docs = None
        emb_index = None
        
        # Always try to load BM25 index (for hybrid retrieval)
        bm25_index_dir = memory_graph_dir.parent / "bm25_index"
        bm25_index_file = bm25_index_dir / f"memory_graph_bm25_index_conv_{conv_id}.pkl"
        if bm25_index_file.exists():
            try:
                with open(bm25_index_file, "rb") as f:
                    index_data = pickle.load(f)
                bm25 = index_data["bm25"]
                docs = index_data["docs"]
            except (EOFError, pickle.UnpicklingError) as e:
                console.print(f"  [red][!] Conversation {conv_id}: BM25 index file corrupted ({e}), please re-run stage 3[/red]")
        
        # If vector retrieval is enabled, load vector index
        retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
        need_emb = retrieval_type in ('vector', 'rrf')
        if need_emb:
            emb_index_dir = memory_graph_dir.parent / "vectors"
            emb_index_file = emb_index_dir / f"memory_graph_embedding_index_conv_{conv_id}.pkl"
            if emb_index_file.exists():
                try:
                    with open(emb_index_file, "rb") as f:
                        emb_index = pickle.load(f)
                except (EOFError, pickle.UnpicklingError) as e:
                    console.print(f"  [red][!] Conversation {conv_id}: Vector index file corrupted ({e}), please re-run stage 3[/red]")
            else:
                console.print(f"  [yellow][!] Conversation {conv_id}: Vector index file not found, using BM25 only[/yellow]")
        
        # Check if at least one index is available
        if bm25 is None and emb_index is None:
            console.print(f"  [yellow][!] Conversation {conv_id}: No available index found[/yellow]")
            return (conv_id_str, [])
        
        # === Perform hierarchical retrieval for each question ===
        # L2L3 associative-recall: load the per-qa top-K map once per conv
        # (it's a single global cached dict; this lookup is O(1)).
        assoc_map = _load_l2l3_assoc_topk()

        # L1 trigger recaller: per-conv artefact, loaded once (env-gated).
        # Returns None when T_MEM_L1_TRIGGER_ENABLED is unset or artefacts
        # are missing — hierarchical_retrieval then runs baseline Layer-3.
        l1_recaller = _load_l1_trigger_recaller(int(conv_id), memory_graph_dir)

        results_for_conv = []
        for qa_pair in conversation_data["qa"]:
            question = qa_pair.get("question")
            if not question:
                continue
            
            # Skip category 5 questions
            if qa_pair.get("category") == 5:
                continue
            
            # Resolve L2L3 associative ids for this (conv_id, question).
            # Missing entry -> None -> hierarchical_retrieval zero-overhead
            # fallback to baseline behavior.
            assoc_ids: Optional[Set[str]] = None
            if assoc_map is not None:
                _ids = assoc_map.get((int(conv_id), question))
                if _ids:
                    assoc_ids = set(_ids)

            # Execute hierarchical retrieval
            hierarchical_results = hierarchical_retrieval(
                query=question,
                memory_graph=memory_graph,
                bm25=bm25,
                docs=docs,
                emb_index=emb_index,
                embedding_provider=embedding_provider,
                reranker_provider=reranker_provider,
                config=config,
                l2l3_assoc_ids=assoc_ids,
                l1_trigger_recaller=l1_recaller,
            )
            
            # Format results (fixed: scene + item only, topic never emitted)
            context_str = format_hierarchical_results(
                results=hierarchical_results,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            )
            
            # [BENCHMARK ADD-ON] trace retrieved scenes/items -> original turn ids
            traced = _extract_retrieved_dia_ids(hierarchical_results, memory_graph)

            # Save results
            results_for_conv.append({
                "query": question,
                "context": context_str,
                "hierarchical_results": {
                    "topics_count": len(hierarchical_results.get("topics", [])),
                    "scenes_count": len(hierarchical_results.get("scenes", [])),
                    "items_count": len(hierarchical_results.get("items", []))
                },
                "retrieval_log": hierarchical_results.get("retrieval_log", {}),
                # ---- benchmark-only enrichment (does not affect QA) ----
                "retrieved_turn_ids": traced["retrieved_turn_ids"],
                "retrieved_scene_ids": traced["retrieved_scene_ids"],
                "retrieved_item_ids": traced["retrieved_item_ids"],
                "turn_ids_from_scenes": traced["turn_ids_from_scenes"],
                "turn_ids_from_items": traced["turn_ids_from_items"],
            })
        
            # Call progress callback
            if progress_callback:
                progress_callback()
        
        return (conv_id_str, results_for_conv)
        
    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: Retrieval failed - {e}[/red]")
        import traceback
        traceback.print_exc()
        return (f"locomo_exp_user_{conv_id}", [])


async def main():
    """Main function: execute batch memory-graph hierarchical retrieval in parallel"""
    # === Configuration ===
    config = ExperimentConfig()

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 6: Memory Graph Retrieval[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Retrieval mode: keyword (BM25 only), vector (vector only), rrf (fusion)
    retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
    retrieval_mode_display = {
        'rrf': '[green]RRF Hybrid Retrieval (BM25 + Vector)[/green]',
        'vector': '[cyan]Vector Retrieval (Vector)[/cyan]',
        'keyword': '[yellow]Keyword Retrieval (BM25)[/yellow]',
    }.get(retrieval_type, f'[red]Unknown mode: {retrieval_type}[/red]')
    console.print(f"[bold]Retrieval mode:[/bold] {retrieval_mode_display}")
    
    # Index directory
    index_dir = config.vectors_dir()

    # Memory graph data directory
    memory_graph_dir = config.memory_graph_dir()

    # Output directory
    save_dir = config.experiment_dir()
    results_output_path = save_dir / "search_results.json"
    
    # Concurrency settings: 14 convs in parallel. Stage 6 is pure
    # embedding + BM25 + rerank HTTP (no LLM), so this only speeds up
    # retrieval and does not affect LLM peak concurrency.
    max_concurrent_tasks = 14
    
    # Dataset path
    dataset_path = Path(config.dataset_path)
    
    # Initialize services
    embedding_provider = None
    need_emb = retrieval_type in ('vector', 'rrf')
    if need_emb:
        embedding_provider = EmbeddingProvider(
            base_url=config.embedding_config["base_url"],
            model_name=config.embedding_config["model_name"],
            max_retries=config.embedding_max_retries
        )
    
    reranker_provider = None
    if config.use_reranker:
        reranker_provider = RerankerProvider(
            base_url=config.reranker_config["base_url"],
            model_name=config.reranker_config["model_name"],
            max_retries=config.reranker_max_retries
        )
        console.print(f"[bold green]Reranker:[/bold green] {reranker_provider}")
    else:
        console.print("[bold yellow]Reranker: DISABLED[/bold yellow]")
    
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}\n")
    
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Load dataset
    console.print(f"[bold]Loading dataset:[/bold] {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    console.print(f"[bold]Number of conversations:[/bold] {len(dataset)}\n")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_with_semaphore(conv_id: int, conversation_data: Dict, task_id: int, progress: Progress, query_count: int):
        """Processing function with semaphore-based concurrency control"""
        async with semaphore:
            progress.start_task(task_id)
            progress.update(task_id, status="Processing")

            # Get the event loop in the main coroutine (before entering the thread pool)
            main_loop = asyncio.get_running_loop()
            
            # Create thread-safe progress callback (capturing main_loop via closure)
            def progress_callback():
                # Use call_soon_threadsafe to ensure thread safety
                main_loop.call_soon_threadsafe(
                    progress.advance, task_id, 1
                )
            
            # Execute retrieval in thread pool (since retrieval involves CPU-intensive operations)
            result = await main_loop.run_in_executor(
                None,
                process_single_conversation_retrieval,
                conv_id,
                conversation_data,
                config,
                memory_graph_dir,
                index_dir,
                embedding_provider,
                reranker_provider,
                progress_callback
            )
            
            conv_id_str, results_for_conv = result
            
            # Ensure progress bar is completed
            progress.update(task_id, completed=query_count, status=f"[green]Done[/green]")
            
            return result
    
    # Create progress bar for parallel processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        # Create tasks for each conversation and count queries
        tasks = []
        for i, conversation_data in enumerate(dataset):
            query_count = get_query_count(conversation_data)
            task_id = progress.add_task(
                f"[cyan]Conv {i}[/cyan]",
                total=query_count if query_count > 0 else 1,
                status="Waiting",
                start=False
            )
            tasks.append((i, conversation_data, task_id, query_count))
        
        # Parallel processing
        coroutines = [
            process_with_semaphore(conv_id, conv_data, task_id, progress, query_count)
            for conv_id, conv_data, task_id, query_count in tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Organize results
    all_search_results = {}
    for result in results:
        if isinstance(result, tuple):
            conv_id_str, results_for_conv = result
            all_search_results[conv_id_str] = results_for_conv
        elif isinstance(result, Exception):
            console.print(f"[red][X] Processing exception: {result}[/red]")

    # === Save all results ===
    console.print(f"\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print(f"[bold]Saving retrieval results to:[/bold] {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)

    # === Save retrieval logs separately for analysis ===
    retrieval_logs_path = save_dir / "retrieval_logs.json"
    all_logs = {}
    for conv_id, items in all_search_results.items():
        all_logs[conv_id] = [
            {"query": item.get("query", ""), "retrieval_log": item.get("retrieval_log", {})}
            for item in items
        ]
    with open(retrieval_logs_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    console.print(f"[bold]Saving retrieval logs to:[/bold] {retrieval_logs_path}")

    console.print(f"[bold green][SUCCESS] Memory-graph hierarchical retrieval completed![/bold green]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())

