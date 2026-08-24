"""Entity/Bridge Trigger Recaller: tri-view nanmax cosine + hard gate (default 0.85).
Per-trigger score = nanmax(cos(q, concept), cos(q, bridge), cos(q, joint)); top-K then
items attached to each survivor (already filtered to conf>=0.70 at build time).
Opt-in only (default OFF if caller never instantiates / passes this recaller)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from T_mem.index.trigger_index import TriggerGraph


@dataclass
class RecalledItem:
    """One memory item surfaced by the Entity/Bridge trigger channel."""

    item_id: str
    via_entity_concept: str = ""
    trigger_quality: float = 0.0
    top1_cosine: float = 0.0
    via_trigger_cos: float = 0.0
    via_confidence: float = 0.0


@dataclass
class TriggerHit:
    """Metadata for one trigger that survived and contributed items."""

    trigger_id: str
    level: int
    concept: str
    cosine: float


@dataclass
class TriggerRecallResult:
    """What the recaller returns per query."""

    triggered: bool
    reason: str
    top1_trigger_id: str = ""
    top1_trigger_level: int = 0
    top1_cosine: float = 0.0
    items: List[RecalledItem] = field(default_factory=list)
    triggers: List[TriggerHit] = field(default_factory=list)

    @property
    def item_ids(self) -> List[str]:
        return [it.item_id for it in self.items]


class TriggerRecaller:
    """Entity/Bridge tri-view nanmax recaller with a hard cosine gate."""

    def __init__(
        self,
        graph: TriggerGraph,
        trigger_ids: List[str],
        trigger_embeddings: np.ndarray,
        trigger_levels: List[int],
        trigger_embeddings_bridge: Optional[np.ndarray] = None,
        trigger_embeddings_joint: Optional[np.ndarray] = None,
    ):
        self.graph = graph
        self.trigger_ids = trigger_ids
        self.trigger_embeddings = trigger_embeddings
        self.trigger_levels = trigger_levels
        self.trigger_embeddings_bridge = trigger_embeddings_bridge
        self.trigger_embeddings_joint = trigger_embeddings_joint

        self._doc_norms = self._safe_norms(trigger_embeddings)
        self._doc_norms_bridge = (
            self._safe_norms(trigger_embeddings_bridge)
            if trigger_embeddings_bridge is not None else None
        )
        self._doc_norms_joint = (
            self._safe_norms(trigger_embeddings_joint)
            if trigger_embeddings_joint is not None else None
        )

        self._id_to_row = {tid: i for i, tid in enumerate(trigger_ids)}

    @staticmethod
    def _safe_norms(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1)
        is_nan_row = np.isnan(mat).any(axis=1)
        safe = np.where((norms == 0) & (~is_nan_row), 1e-9, norms)
        return safe

    @classmethod
    def from_dir(
        cls, trigger_dir: Path | str, conv_id: int
    ) -> Optional["TriggerRecaller"]:
        """Load graph+tri-view embeddings; returns None when either artefact is missing."""
        base = Path(trigger_dir)
        graph_path = base / f"entity_bridge_graph_conv_{conv_id}.json"
        emb_path = base / f"entity_bridge_embeddings_conv_{conv_id}.npz"

        if not graph_path.exists() or not emb_path.exists():
            return None

        graph = TriggerGraph.load(graph_path)

        ids, emb_c, levels, emb_b, emb_j = TriggerGraph.load_embeddings_triview(
            emb_path
        )
        return cls(
            graph=graph,
            trigger_ids=list(ids),
            trigger_embeddings=emb_c,
            trigger_levels=list(levels),
            trigger_embeddings_bridge=emb_b,
            trigger_embeddings_joint=emb_j,
        )

    def _cos_one_view(
        self,
        q: np.ndarray,
        q_norm: float,
        mat: Optional[np.ndarray],
        norms: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if mat is None or norms is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return (mat @ q) / (norms * q_norm)

    def recall(
        self,
        query_embedding: np.ndarray,
        max_top_triggers: int = 10,
        min_cosine_gate: float = 0.85,
    ) -> TriggerRecallResult:
        """Tri-view nanmax top-K recall; min_cosine_gate default 0.85 (HIGH, additive-safe)."""
        if len(self.trigger_ids) == 0:
            return TriggerRecallResult(triggered=False, reason="no triggers loaded")

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return TriggerRecallResult(triggered=False, reason="zero query vector")

        cos_c = self._cos_one_view(
            q, q_norm, self.trigger_embeddings, self._doc_norms
        )
        cos_b = self._cos_one_view(
            q, q_norm, self.trigger_embeddings_bridge, self._doc_norms_bridge
        )
        cos_j = self._cos_one_view(
            q, q_norm, self.trigger_embeddings_joint, self._doc_norms_joint
        )

        stack_list = [cos_c]
        if cos_b is not None:
            stack_list.append(cos_b)
        if cos_j is not None:
            stack_list.append(cos_j)
        stacked = np.vstack(stack_list)
        with np.errstate(invalid="ignore"):
            scores = np.nanmax(stacked, axis=0)
        scores = np.where(np.isnan(scores), -np.inf, scores)

        levels_arr = np.asarray(self.trigger_levels, dtype=np.int32)

        order = np.argsort(-scores, kind="stable")
        top_k = max(1, int(max_top_triggers))
        picked: List[tuple] = []
        n_dropped_by_gate = 0
        for idx in order[:top_k]:
            s = float(scores[idx])
            if s == -np.inf:
                continue
            if s < min_cosine_gate:
                n_dropped_by_gate += 1
                continue
            picked.append((s, int(idx), int(levels_arr[idx])))

        if not picked:
            best_score = float(scores.max()) if scores.size else 0.0
            best_idx = int(np.argmax(scores)) if scores.size else -1
            return TriggerRecallResult(
                triggered=False,
                reason=(
                    f"no trigger passed gate={min_cosine_gate:.2f} "
                    f"(best={best_score:.3f}, n_dropped_by_gate={n_dropped_by_gate})"
                ),
                top1_trigger_id=(self.trigger_ids[best_idx] if best_idx >= 0 else ""),
                top1_trigger_level=1,
                top1_cosine=best_score,
            )

        items_out: List[RecalledItem] = []
        seen_item_ids: set = set()
        hit_meta: List[TriggerHit] = []
        expansion_reasons: List[str] = []

        for cos, idx, lvl in picked:
            tid = self.trigger_ids[idx]

            eb = self.graph.entity_bridge_triggers.get(tid)
            if eb is None or not eb.item_confidences:
                expansion_reasons.append(
                    f"trigger {tid} missing/no-items (score={cos:.3f})"
                )
                continue

            item_list = sorted(
                eb.item_confidences.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )

            expanded_here = 0
            for iid, conf in item_list:
                if iid in seen_item_ids:
                    continue
                seen_item_ids.add(iid)
                items_out.append(
                    RecalledItem(
                        item_id=iid,
                        via_entity_concept=eb.concept,
                        trigger_quality=eb.quality,
                        top1_cosine=cos,
                        via_trigger_cos=cos,
                        via_confidence=float(conf),
                    )
                )
                expanded_here += 1

            hit_meta.append(
                TriggerHit(
                    trigger_id=tid, level=1, concept=eb.concept, cosine=cos
                )
            )
            expansion_reasons.append(
                f"trigger '{eb.concept[:40]}' score={cos:.3f} (+{expanded_here} items)"
            )

        if not items_out:
            return TriggerRecallResult(
                triggered=False,
                reason=(
                    "all survivors empty-after-expansion: "
                    + " | ".join(expansion_reasons)
                ),
                top1_trigger_id=self.trigger_ids[picked[0][1]],
                top1_trigger_level=int(picked[0][2]),
                top1_cosine=float(picked[0][0]),
            )

        top_hit = hit_meta[0] if hit_meta else None
        return TriggerRecallResult(
            triggered=True,
            reason=(
                f"{len(hit_meta)} trigger(s) fired (gate={min_cosine_gate:.2f}, "
                f"dropped_by_gate={n_dropped_by_gate}): "
                + " | ".join(expansion_reasons)
            ),
            top1_trigger_id=(top_hit.trigger_id if top_hit else ""),
            top1_trigger_level=(top_hit.level if top_hit else 0),
            top1_cosine=(top_hit.cosine if top_hit else 0.0),
            items=items_out,
            triggers=hit_meta,
        )
