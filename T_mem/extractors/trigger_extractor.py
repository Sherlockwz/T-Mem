"""Entity/Bridge trigger extractor (3 phases): per-item extraction -> rapidfuzz dedup -> per-trigger confidence filter.
LLM calls are async, gated by max_concurrent; per-item retries up to extract_retries before skip.
See T_mem.prompts.trigger_prompts for the Entity/Bridge/Scene/Horizon paper-vs-code naming mapping."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import json_repair  # type: ignore

    _HAS_JSON_REPAIR = True
except ImportError:  # pragma: no cover
    _HAS_JSON_REPAIR = False

from T_mem.prompts.trigger_prompts import (
    ENTITY_BRIDGE_TRIGGER_PROMPT,
)
from T_mem.index.trigger_index import (
    EntityBridgeTrigger,
    TriggerGraph,
    normalize_concept,
)


@dataclass
class ExtractorConfig:
    """Phase-level hyperparameters for the trigger extractor."""

    entity_bridge_count_per_item: int = 5
    entity_bridge_dedup_ratio: float = 0.90
    item_conf_threshold: float = 0.70

    max_concurrent: int = 14
    extract_retries: int = 3


@dataclass
class ExtractionStats:
    """Per-phase counters for debug/log."""

    n_items_input: int = 0
    n_entity_bridge_raw: int = 0
    n_entity_bridge_after_dedup: int = 0
    n_entity_bridge_after_filter: int = 0
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_items_input": self.n_items_input,
            "n_entity_bridge_raw": self.n_entity_bridge_raw,
            "n_entity_bridge_after_dedup": self.n_entity_bridge_after_dedup,
            "n_entity_bridge_after_filter": self.n_entity_bridge_after_filter,
            "n_failures": len(self.failures),
        }


def _parse_json_robust(text: str) -> Any:
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        if _HAS_JSON_REPAIR:
            return json_repair.loads(t)  # type: ignore[return-value]
        raise


def _clean_concept(s: Optional[str]) -> str:
    return (s or "").strip()


class TriggerExtractor:
    """Entity/Bridge trigger extractor over a batch of memory items; returns a TriggerGraph (no embeddings)."""

    def __init__(
        self,
        llm_provider,
        config: Optional[ExtractorConfig] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm_provider
        self.config = config or ExtractorConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        self._log = logger or (lambda msg: print(msg, flush=True))

    async def _extract_entity_bridge_for_item(
        self, item: Dict[str, Any]
    ) -> Tuple[str, List[EntityBridgeTrigger]]:
        item_id = item["item_id"]
        content = item.get("content", "") or ""
        temporal = item.get("temporal", "") or "Not specified"

        prompt = ENTITY_BRIDGE_TRIGGER_PROMPT.format(
            item_content=content,
            item_temporal=temporal,
            trigger_count=self.config.entity_bridge_count_per_item,
        )

        last_err: Optional[str] = None
        for attempt in range(1, self.config.extract_retries + 1):
            try:
                async with self._sem:
                    resp = await self.llm.generate(
                        prompt,
                        response_format={"type": "json_object"},
                        call_site="entity_bridge_trigger.item",
                    )
                data = _parse_json_robust(resp)
                triggers = self._validate_parse_entity_bridge(data, item_id)
                if not triggers:
                    raise ValueError("no valid triggers parsed")
                return item_id, triggers
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < self.config.extract_retries:
                    await asyncio.sleep(0.5 * attempt)
                else:
                    self._log(
                        f"  [ENTITY_BRIDGE FAIL] item={item_id[:20]}... after "
                        f"{self.config.extract_retries} tries: {last_err}"
                    )

        return item_id, []

    def _validate_parse_entity_bridge(
        self, data: Any, item_id: str
    ) -> List[EntityBridgeTrigger]:
        if not isinstance(data, dict):
            raise ValueError("top-level not a dict")
        triggers = data.get("triggers")
        if not isinstance(triggers, list) or not triggers:
            raise ValueError("'triggers' missing or empty")

        out: List[EntityBridgeTrigger] = []
        seen_norm: set = set()
        for entry in triggers:
            if not isinstance(entry, dict):
                continue
            concept = _clean_concept(entry.get("concept"))
            if not concept:
                continue
            norm = normalize_concept(concept)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)

            try:
                conf = float(entry.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            patterns = entry.get("activation_patterns") or []
            if not isinstance(patterns, list):
                patterns = []
            patterns = [
                str(p).strip() for p in patterns if isinstance(p, (str, int, float))
            ][:3]

            eb = EntityBridgeTrigger(
                concept=concept,
                concept_norm=norm,
                bridge=_clean_concept(entry.get("bridge")),
                activation_patterns=patterns,
                item_confidences={item_id: conf},
            )
            out.append(eb)
        return out

    async def extract(
        self,
        items: List[Dict[str, Any]],
        conv_id: Optional[int] = None,
    ) -> Tuple[TriggerGraph, ExtractionStats]:
        """Run Phase 1~3 and return (graph, stats)."""
        stats = ExtractionStats(n_items_input=len(items))
        graph = TriggerGraph(conv_id=conv_id)

        self._log(f"\n[Phase 1] Extracting entity/bridge triggers from {len(items)} items...")
        tasks = [self._extract_entity_bridge_for_item(it) for it in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                stats.failures.append(f"phase1: {res!r}")
                continue
            _item_id, eb_list = res
            for eb in eb_list:
                graph.entity_bridge_triggers[eb.id] = eb
                stats.n_entity_bridge_raw += 1

        self._log(f"[Phase 1] Done. {stats.n_entity_bridge_raw} raw entity/bridge triggers")

        self._log(
            f"[Phase 2] Entity/bridge dedup (ratio>={self.config.entity_bridge_dedup_ratio})..."
        )
        eb_redirect = graph.dedup_entity_bridge(similarity_threshold=self.config.entity_bridge_dedup_ratio)
        stats.n_entity_bridge_after_dedup = len(graph.entity_bridge_triggers)
        self._log(
            f"[Phase 2] Done. {stats.n_entity_bridge_raw} → {stats.n_entity_bridge_after_dedup} "
            f"({len(eb_redirect)} merged)"
        )

        self._log(
            f"[Phase 3] Filter items per trigger (conf>={self.config.item_conf_threshold})..."
        )
        graph.filter_entity_bridge_items(
            conf_threshold=self.config.item_conf_threshold,
        )
        dropped_no_item = [eb for eb in graph.entity_bridge_triggers.values() if not eb.item_confidences]
        for eb in dropped_no_item:
            del graph.entity_bridge_triggers[eb.id]
        stats.n_entity_bridge_after_filter = len(graph.entity_bridge_triggers)
        self._log(
            f"[Phase 3] Done. {stats.n_entity_bridge_after_dedup} → {stats.n_entity_bridge_after_filter} "
            f"(dropped {len(dropped_no_item)} triggers with no high-conf items)"
        )

        self._log(f"\n[Extractor] Final stats: {graph.stats()}")
        return graph, stats
