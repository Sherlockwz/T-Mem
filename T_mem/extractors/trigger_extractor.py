"""L1 Trigger extractor (3 phases): per-item extraction -> rapidfuzz dedup -> per-L1 confidence filter.
LLM calls are async, gated by max_concurrent; per-item retries up to extract_retries before skip.
See T_mem.prompts.trigger_prompts for the L1/L2/L3 paper-vs-code naming mapping."""

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
    L1_TRIGGER_PROMPT,
)
from T_mem.index.trigger_index import (
    L1Trigger,
    TriggerGraph,
    normalize_concept,
)


@dataclass
class ExtractorConfig:
    """Phase-level hyperparameters for the trigger extractor."""

    l1_count_per_item: int = 5
    l1_dedup_ratio: float = 0.90
    item_conf_threshold: float = 0.70

    max_concurrent: int = 14
    extract_retries: int = 3


@dataclass
class ExtractionStats:
    """Per-phase counters for debug/log."""

    n_items_input: int = 0
    n_l1_raw: int = 0
    n_l1_after_dedup: int = 0
    n_l1_after_filter: int = 0
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_items_input": self.n_items_input,
            "n_l1_raw": self.n_l1_raw,
            "n_l1_after_dedup": self.n_l1_after_dedup,
            "n_l1_after_filter": self.n_l1_after_filter,
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
    """L1 trigger extractor over a batch of memory items; returns a TriggerGraph (no embeddings)."""

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

    async def _extract_l1_for_item(
        self, item: Dict[str, Any]
    ) -> Tuple[str, List[L1Trigger]]:
        item_id = item["item_id"]
        content = item.get("content", "") or ""
        temporal = item.get("temporal", "") or "Not specified"

        prompt = L1_TRIGGER_PROMPT.format(
            item_content=content,
            item_temporal=temporal,
            trigger_count=self.config.l1_count_per_item,
        )

        last_err: Optional[str] = None
        for attempt in range(1, self.config.extract_retries + 1):
            try:
                async with self._sem:
                    resp = await self.llm.generate(
                        prompt,
                        response_format={"type": "json_object"},
                        call_site="l1trigger.item",
                    )
                data = _parse_json_robust(resp)
                triggers = self._validate_parse_l1(data, item_id)
                if not triggers:
                    raise ValueError("no valid triggers parsed")
                return item_id, triggers
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < self.config.extract_retries:
                    await asyncio.sleep(0.5 * attempt)
                else:
                    self._log(
                        f"  [L1 FAIL] item={item_id[:20]}... after "
                        f"{self.config.extract_retries} tries: {last_err}"
                    )

        return item_id, []

    def _validate_parse_l1(
        self, data: Any, item_id: str
    ) -> List[L1Trigger]:
        if not isinstance(data, dict):
            raise ValueError("top-level not a dict")
        triggers = data.get("triggers")
        if not isinstance(triggers, list) or not triggers:
            raise ValueError("'triggers' missing or empty")

        out: List[L1Trigger] = []
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

            l1 = L1Trigger(
                concept=concept,
                concept_norm=norm,
                bridge=_clean_concept(entry.get("bridge")),
                activation_patterns=patterns,
                item_confidences={item_id: conf},
            )
            out.append(l1)
        return out

    async def extract(
        self,
        items: List[Dict[str, Any]],
        conv_id: Optional[int] = None,
    ) -> Tuple[TriggerGraph, ExtractionStats]:
        """Run Phase 1~3 and return (graph, stats)."""
        stats = ExtractionStats(n_items_input=len(items))
        graph = TriggerGraph(conv_id=conv_id)

        self._log(f"\n[Phase 1] Extracting L1 triggers from {len(items)} items...")
        tasks = [self._extract_l1_for_item(it) for it in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                stats.failures.append(f"phase1: {res!r}")
                continue
            _item_id, l1_list = res
            for l1 in l1_list:
                graph.l1_triggers[l1.id] = l1
                stats.n_l1_raw += 1

        self._log(f"[Phase 1] Done. {stats.n_l1_raw} raw L1 triggers")

        self._log(
            f"[Phase 2] L1 dedup (ratio>={self.config.l1_dedup_ratio})..."
        )
        l1_redirect = graph.dedup_l1(similarity_threshold=self.config.l1_dedup_ratio)
        stats.n_l1_after_dedup = len(graph.l1_triggers)
        self._log(
            f"[Phase 2] Done. {stats.n_l1_raw} → {stats.n_l1_after_dedup} "
            f"({len(l1_redirect)} merged)"
        )

        self._log(
            f"[Phase 3] Filter items per L1 (conf>={self.config.item_conf_threshold})..."
        )
        graph.filter_l1_items(
            conf_threshold=self.config.item_conf_threshold,
        )
        dropped_no_item = [l1 for l1 in graph.l1_triggers.values() if not l1.item_confidences]
        for l1 in dropped_no_item:
            del graph.l1_triggers[l1.id]
        stats.n_l1_after_filter = len(graph.l1_triggers)
        self._log(
            f"[Phase 3] Done. {stats.n_l1_after_dedup} → {stats.n_l1_after_filter} "
            f"(dropped {len(dropped_no_item)} L1 with no high-conf items)"
        )

        self._log(f"\n[Extractor] Final stats: {graph.stats()}")
        return graph, stats
