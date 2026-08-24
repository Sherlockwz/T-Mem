"""Entity/Bridge Trigger embedder: tri-view BGE embeddings (concept / bridge / joint).
At recall: cos(q, each view) -> nanmax. Empty-bridge triggers degenerate to concept-only
(bridge/joint slots become NaN sentinels, skipped by the NaN-aware max)."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .trigger_index import TriggerGraph


def embed_trigger_graph_triview(
    graph: TriggerGraph,
    embedding_provider,
    batch_size: int = 32,
    logger=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returns (emb_concept, emb_bridge, emb_joint); empty-bridge triggers are absent from bridge/joint."""
    log = logger or (lambda m: print(m, flush=True))

    concept_items: List[Tuple[str, str]] = []
    bridge_items: List[Tuple[str, str]] = []
    joint_items: List[Tuple[str, str]] = []

    for tid, eb in graph.entity_bridge_triggers.items():
        concept_items.append((tid, eb.to_text_for_embedding()))
        bridge_text = (eb.bridge or "").strip()
        if bridge_text:
            bridge_items.append((tid, bridge_text))
            joint_items.append(
                (tid, f"{eb.concept.strip()} . {bridge_text}")
            )

    n_c, n_b, n_j = len(concept_items), len(bridge_items), len(joint_items)
    log(
        f"[Embedder/tri] entity_bridge={len(graph.entity_bridge_triggers)} "
        f"→ texts to embed: concept={n_c}, bridge={n_b}, joint={n_j}"
    )

    start = time.time()

    def _run_batch(items: List[Tuple[str, str]], label: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        if not items:
            return out
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            ids = [tid for tid, _ in batch]
            texts = [txt for _, txt in batch]
            t0 = time.time()
            vecs = embedding_provider.embed(texts)
            dt = time.time() - t0
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"embedding provider returned {len(vecs)} vecs for "
                    f"{len(batch)} texts (view={label})"
                )
            for tid, vec in zip(ids, vecs):
                out[tid] = np.asarray(vec, dtype=np.float32)
            if (i // batch_size) % 20 == 0 or (i + batch_size) >= len(items):
                progress = min(i + batch_size, len(items))
                log(
                    f"[Embedder/tri/{label}] {progress}/{len(items)} "
                    f"({progress * 100 / len(items):.1f}%) "
                    f"last batch {len(batch)} in {dt:.1f}s"
                )
        return out

    emb_concept = _run_batch(concept_items, "concept")
    emb_bridge = _run_batch(bridge_items, "bridge")
    emb_joint = _run_batch(joint_items, "joint")

    log(
        f"[Embedder/tri] done in {time.time() - start:.1f}s — "
        f"concept={len(emb_concept)}, bridge={len(emb_bridge)}, joint={len(emb_joint)}"
    )
    return emb_concept, emb_bridge, emb_joint
