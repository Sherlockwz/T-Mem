"""Trigger graph data structures (Entity/Bridge): nodes, edges, dedup, save/load.
EntityBridgeTrigger holds concept (Entity Trigger route) + bridge (Bridge Trigger route) + item_confidences.
See T_mem.prompts.trigger_prompts for the Entity/Bridge/Scene/Horizon paper-vs-code naming mapping."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # optional — only used in save_embeddings/load_embeddings
except ImportError:  # pragma: no cover
    np = None  # type: ignore


class EntityBridgeTrigger:
    """Item-level trigger covering both Entity (Q I) and Bridge (Q II) routes.

    Connected to one or more memory items via (item_id, confidence) pairs.
    The trigger's `quality` is max(confidence over its items).
    """

    __slots__ = (
        "id",
        "concept",
        "concept_norm",
        "bridge",
        "activation_patterns",
        "item_confidences",
        "created_at",
    )

    def __init__(
        self,
        concept: str,
        concept_norm: str,
        bridge: str = "",
        activation_patterns: Optional[List[str]] = None,
        item_confidences: Optional[Dict[str, float]] = None,
        trigger_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.id = trigger_id or f"eb_{uuid.uuid4().hex[:12]}"
        self.concept = concept
        self.concept_norm = concept_norm
        self.bridge = bridge
        self.activation_patterns = list(activation_patterns or [])
        self.item_confidences: Dict[str, float] = dict(item_confidences or {})
        self.created_at = created_at or datetime.now().isoformat()

    @property
    def item_ids(self) -> List[str]:
        return list(self.item_confidences.keys())

    @property
    def quality(self) -> float:
        if not self.item_confidences:
            return 0.0
        return max(self.item_confidences.values())

    def add_item(self, item_id: str, confidence: float):
        prev = self.item_confidences.get(item_id, -1.0)
        if confidence > prev:
            self.item_confidences[item_id] = float(confidence)

    def merge_from(self, other: "EntityBridgeTrigger"):
        """Absorb another trigger into this one (used during global dedup); first-seen concept wins."""
        for iid, conf in other.item_confidences.items():
            self.add_item(iid, conf)
        seen = set(self.activation_patterns)
        for p in other.activation_patterns:
            if p not in seen:
                self.activation_patterns.append(p)
                seen.add(p)
        if len(other.bridge) > len(self.bridge):
            self.bridge = other.bridge

    def filter_items(self, conf_threshold: float) -> None:
        kept = [
            (iid, c) for iid, c in self.item_confidences.items() if c >= conf_threshold
        ]
        kept.sort(key=lambda kv: kv[1], reverse=True)
        self.item_confidences = dict(kept)

    def to_text_for_embedding(self) -> str:
        # Env-controlled mode: "concept_only" (default) or "concept_bridge".
        import os
        mode = os.environ.get("TRIGGER_EMBED_TEXT_MODE", "concept_only").strip().lower()
        parts: List[str] = [self.concept]
        if mode == "concept_bridge":
            if self.bridge:
                parts.append(self.bridge)
        else:
            if self.activation_patterns:
                parts.extend(self.activation_patterns)
        return " . ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": 1,
            "concept": self.concept,
            "concept_norm": self.concept_norm,
            "bridge": self.bridge,
            "activation_patterns": self.activation_patterns,
            "item_confidences": self.item_confidences,
            "quality": round(self.quality, 4),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityBridgeTrigger":
        return cls(
            concept=d["concept"],
            concept_norm=d.get("concept_norm", d["concept"].lower().strip()),
            bridge=d.get("bridge", ""),
            activation_patterns=d.get("activation_patterns", []),
            item_confidences=d.get("item_confidences", {}),
            trigger_id=d.get("id"),
            created_at=d.get("created_at"),
        )


class TriggerGraph:
    """Container of all Entity/Bridge triggers for one conversation; stores item IDs (not content)."""

    def __init__(
        self,
        conv_id: Optional[int] = None,
        entity_bridge_triggers: Optional[Dict[str, EntityBridgeTrigger]] = None,
    ):
        self.conv_id = conv_id
        self.entity_bridge_triggers: Dict[str, EntityBridgeTrigger] = dict(entity_bridge_triggers or {})

    def dedup_entity_bridge(self, similarity_threshold: float = 0.9) -> Dict[str, str]:
        """Dedup Entity/Bridge triggers via normalized + rapidfuzz match; returns old_id -> canonical_id map."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            fuzz = None  # type: ignore

        canonical: Dict[str, EntityBridgeTrigger] = {}
        redirect: Dict[str, str] = {}

        # CRITICAL: iterate in deterministic created_at order so canonical pick is stable
        ordered = sorted(self.entity_bridge_triggers.values(), key=lambda t: t.created_at)

        for eb in ordered:
            merged = False
            norm = eb.concept_norm

            if norm in canonical:
                target = canonical[norm]
                if target.id != eb.id:
                    target.merge_from(eb)
                    redirect[eb.id] = target.id
                merged = True
                continue

            if fuzz is not None:
                best_key = None
                best_score = 0.0
                for ckey in canonical.keys():
                    score = fuzz.ratio(norm, ckey) / 100.0
                    if score > best_score:
                        best_score = score
                        best_key = ckey
                if (
                    best_key is not None
                    and best_score >= similarity_threshold
                    and canonical[best_key].id != eb.id
                ):
                    canonical[best_key].merge_from(eb)
                    redirect[eb.id] = canonical[best_key].id
                    merged = True

            if not merged:
                canonical[norm] = eb

        self.entity_bridge_triggers = {t.id: t for t in canonical.values()}
        return redirect

    def filter_entity_bridge_items(self, conf_threshold: float) -> None:
        # NOTE: keep triggers that lost all items — their activation_patterns may still
        # contribute at recall time.
        for eb in self.entity_bridge_triggers.values():
            eb.filter_items(conf_threshold)

    def stats(self) -> Dict[str, Any]:
        n_entity_bridge = len(self.entity_bridge_triggers)
        trigger_fanout = [len(eb.item_confidences) for eb in self.entity_bridge_triggers.values()]
        unique_items = {
            iid
            for eb in self.entity_bridge_triggers.values()
            for iid in eb.item_confidences.keys()
        }
        return {
            "conv_id": self.conv_id,
            "n_entity_bridge": n_entity_bridge,
            "unique_items_covered": len(unique_items),
            "avg_items_per_entity_bridge": (
                round(sum(trigger_fanout) / max(1, n_entity_bridge), 2) if n_entity_bridge else 0
            ),
            "max_items_per_entity_bridge": max(trigger_fanout) if trigger_fanout else 0,
            "entity_bridge_quality_distribution": _quality_dist(
                [eb.quality for eb in self.entity_bridge_triggers.values()]
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conv_id": self.conv_id,
            "entity_bridge_triggers": {tid: t.to_dict() for tid, t in self.entity_bridge_triggers.items()},
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriggerGraph":
        g = cls(conv_id=data.get("conv_id"))
        g.entity_bridge_triggers = {
            tid: EntityBridgeTrigger.from_dict(d) for tid, d in data.get("entity_bridge_triggers", {}).items()
        }
        return g
    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "TriggerGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_embeddings(
        self,
        path: Path | str,
        embeddings: Dict[str, "np.ndarray"],
    ) -> None:
        """Save trigger id -> embedding as npz (ids, embeddings, levels)."""
        if np is None:
            raise RuntimeError("numpy is required to save embeddings")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ids: List[str] = []
        mats: List["np.ndarray"] = []
        levels: List[int] = []
        for tid, emb in embeddings.items():
            if tid not in self.entity_bridge_triggers:
                continue
            levels.append(1)
            ids.append(tid)
            mats.append(np.asarray(emb, dtype=np.float32))
        if not ids:
            raise ValueError("no embeddings to save")
        matrix = np.stack(mats, axis=0)
        np.savez(
            p,
            ids=np.array(ids, dtype=object),
            embeddings=matrix,
            levels=np.array(levels, dtype=np.int8),
        )

    @classmethod
    def load_embeddings(
        cls, path: Path | str
    ) -> Tuple[List[str], "np.ndarray", List[int]]:
        if np is None:
            raise RuntimeError("numpy is required to load embeddings")
        data = np.load(path, allow_pickle=True)
        ids = list(data["ids"])
        embeddings = data["embeddings"]
        levels = list(data["levels"])
        return ids, embeddings, levels

    @classmethod
    def load_embeddings_with_bridge(
        cls, path: Path | str
    ) -> Tuple[List[str], "np.ndarray", List[int], Optional["np.ndarray"]]:
        """Like :meth:`load_embeddings`, but also returns the optional bridge view.

        Returns ``(ids, emb_concept, levels, emb_bridge_or_None)``.
        """
        if np is None:
            raise RuntimeError("numpy is required to load embeddings")
        data = np.load(path, allow_pickle=True)
        ids = list(data["ids"])
        embeddings = data["embeddings"]
        levels = list(data["levels"])
        emb_bridge = None
        if "embeddings_bridge" in data.files:
            emb_bridge = data["embeddings_bridge"]
            if emb_bridge.shape != embeddings.shape:
                raise ValueError(
                    f"embeddings_bridge shape {emb_bridge.shape} does not "
                    f"match embeddings shape {embeddings.shape} in {path}"
                )
        return ids, embeddings, levels, emb_bridge

    def save_embeddings_triview(
        self,
        path: Path | str,
        emb_concept: Dict[str, "np.ndarray"],
        emb_bridge: Dict[str, "np.ndarray"],
        emb_joint: Dict[str, "np.ndarray"],
    ) -> None:
        """Save concept/bridge/joint per-trigger embedding views into a single npz.

        Rows are aligned across the three views; triggers lacking a bridge or
        joint vector are stored as NaN, which ``nanmax`` collapses back to
        concept-only at recall time.
        """
        if np is None:
            raise RuntimeError("numpy is required to save embeddings")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        ids: List[str] = []
        levels: List[int] = []
        concept_mats: List["np.ndarray"] = []
        bridge_mats: List["np.ndarray"] = []
        joint_mats: List["np.ndarray"] = []

        if not emb_concept:
            raise ValueError("emb_concept is empty, nothing to save")

        first_vec = next(iter(emb_concept.values()))
        dim = int(np.asarray(first_vec).shape[-1])
        nan_row = np.full((dim,), np.nan, dtype=np.float32)

        for tid, emb_c in emb_concept.items():
            if tid not in self.entity_bridge_triggers:
                continue
            ids.append(tid)
            levels.append(1)
            concept_mats.append(np.asarray(emb_c, dtype=np.float32))

            b_vec = emb_bridge.get(tid)
            bridge_mats.append(
                np.asarray(b_vec, dtype=np.float32) if b_vec is not None else nan_row
            )
            j_vec = emb_joint.get(tid)
            joint_mats.append(
                np.asarray(j_vec, dtype=np.float32) if j_vec is not None else nan_row
            )

        if not ids:
            raise ValueError("no embeddings to save after filtering")

        np.savez(
            p,
            ids=np.array(ids, dtype=object),
            embeddings=np.stack(concept_mats, axis=0),
            embeddings_bridge=np.stack(bridge_mats, axis=0),
            embeddings_joint=np.stack(joint_mats, axis=0),
            levels=np.array(levels, dtype=np.int8),
        )

    @classmethod
    def load_embeddings_triview(
        cls, path: Path | str
    ) -> Tuple[
        List[str],
        "np.ndarray",
        List[int],
        Optional["np.ndarray"],
        Optional["np.ndarray"],
    ]:
        """Load all three views. Returns
            (ids, emb_concept, levels, emb_bridge_or_None, emb_joint_or_None).
        """
        if np is None:
            raise RuntimeError("numpy is required to load embeddings")
        data = np.load(path, allow_pickle=True)
        ids = list(data["ids"])
        embeddings = data["embeddings"]
        levels = list(data["levels"])
        emb_bridge = data["embeddings_bridge"] if "embeddings_bridge" in data.files else None
        emb_joint = data["embeddings_joint"] if "embeddings_joint" in data.files else None

        for name, arr in (("embeddings_bridge", emb_bridge), ("embeddings_joint", emb_joint)):
            if arr is not None and arr.shape != embeddings.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match embeddings shape "
                    f"{embeddings.shape} in {path}"
                )
        return ids, embeddings, levels, emb_bridge, emb_joint


def normalize_concept(text: str) -> str:
    """Aggressive normalization for Entity/Bridge trigger concept dedup.

    Lowercase, strip, collapse whitespace, remove common punctuation at edges.
    """
    if not text:
        return ""
    t = text.strip().lower()
    t = " ".join(t.split())
    t = t.strip(".,?!;:\"'`()[]{}<>-—_")
    return t


def _quality_dist(values: List[float]) -> Dict[str, int]:
    """Bucket entity/bridge trigger quality into readable histogram."""
    buckets = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, ">=0.85": 0}
    for v in values:
        if v < 0.5:
            buckets["<0.5"] += 1
        elif v < 0.7:
            buckets["0.5-0.7"] += 1
        elif v < 0.85:
            buckets["0.7-0.85"] += 1
        else:
            buckets[">=0.85"] += 1
    return buckets
