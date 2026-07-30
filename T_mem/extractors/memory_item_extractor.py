"""Topic-based memory-item extractor (single-stage)."""

import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print(
        "Warning: json_repair library is not installed. Standard json parsing will be used. "
        "Install with: pip install json-repair"
    )

from T_mem.utils.logger import get_logger
from T_mem.llm.llm_provider import LLMProvider
from T_mem.types import MemoryItem, Scene, Topic
from T_mem.prompts.memory_item_prompts import ITEM_EXTRACTION_PROMPT

logger = get_logger(__name__)


@dataclass
class MemoryItemExtractResult:
    topic_id: str
    items: List[MemoryItem]
    reasoning: str = ""

    def __post_init__(self):
        self.item_count = len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "item_count": self.item_count,
            "items": [item.to_dict() for item in self.items],
            "reasoning": self.reasoning,
        }

    def get_items_as_text(self) -> List[str]:
        return [item.to_text() for item in self.items]


class MemoryItemExtractor:
    """Topic-based memory-item extractor; emitted items carry scene_ids for downstream graph linking."""

    def __init__(self, llm_provider=LLMProvider, **llm_kwargs):
        self.llm_provider = llm_provider
        self.llm_kwargs = llm_kwargs

    def _format_topic_context(self, topic: Topic) -> Tuple[str, str, str]:
        return topic.topic_id, topic.title, topic.summary

    def _format_scenes_content(self, scenes: List[Scene]) -> str:
        lines = []
        for i, scene in enumerate(scenes):
            lines.append(f"--- Scene {i+1} (ID: scene_{i+1}) ---")

            if scene.subject:
                lines.append(f"Subject: {scene.subject}")
            if scene.summary:
                lines.append(f"Summary: {scene.summary}")
            if scene.scene_description:
                lines.append(f"Scene: {scene.scene_description}")
            if scene.keywords:
                lines.append(f"Keywords: {', '.join(scene.keywords)}")

            if scene.original_data:
                lines.append("Original Content:")
                for j, data in enumerate(scene.original_data):
                    if isinstance(data, dict):
                        if 'content' in data and 'speaker_name' in data:
                            lines.append(
                                f"  [{j+1}] {data.get('speaker_name', 'Unknown')}: "
                                f"{data.get('content', '')}"
                            )
                        elif 'text' in data:
                            lines.append(f"  [{j+1}] {data.get('text', '')}")
                        else:
                            lines.append(f"  [{j+1}] {json.dumps(data, ensure_ascii=False)}")
                    else:
                        lines.append(f"  [{j+1}] {data}")

            lines.append("")

        return "\n".join(lines)

    def _get_reference_time(self, scenes: List[Scene]) -> str:
        if not scenes:
            return "Not specified"

        latest_timestamp = None
        for scene in scenes:
            if scene.timestamp:
                if latest_timestamp is None or scene.timestamp > latest_timestamp:
                    latest_timestamp = scene.timestamp

        if latest_timestamp:
            if isinstance(latest_timestamp, datetime):
                return latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            return str(latest_timestamp)

        return "Not specified"

    # -------- Validation --------

    def _validate_item_extraction(
        self,
        data: Dict[str, Any],
        valid_scene_ids: set,
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if "items" not in data:
            errors.append("Missing required field 'items'")
            return False, errors

        items = data.get("items", [])

        if not isinstance(items, list):
            errors.append("'items' must be a list type")
            return False, errors

        if len(items) == 0:
            errors.append("At least one item must be extracted")
            return False, errors

        item_ids: set = set()
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"Item #{i+1} must be a dict type")
                continue

            required_fields = ["item_id", "content", "scene_ids"]
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item #{i+1} is missing required field '{field}'")
                elif not item[field]:
                    errors.append(f"Item #{i+1} field '{field}' cannot be empty")

            item_id = item.get("item_id", "")
            if item_id in item_ids:
                errors.append(f"Item ID '{item_id}' is duplicated")
            item_ids.add(item_id)

            scene_ids_list = item.get("scene_ids", [])
            if not isinstance(scene_ids_list, list):
                errors.append(f"Item '{item_id}' 'scene_ids' must be a list type")
            else:
                for sc_id in scene_ids_list:
                    if sc_id not in valid_scene_ids:
                        errors.append(f"Item '{item_id}' references non-existent scene '{sc_id}'")

            if "keywords" in item and item["keywords"] is not None:
                if not isinstance(item["keywords"], list):
                    errors.append(f"Item '{item_id}' 'keywords' must be a list type")

            if "query_patterns" in item and item["query_patterns"] is not None:
                if not isinstance(item["query_patterns"], list):
                    errors.append(f"Item '{item_id}' 'query_patterns' must be a list type")

        return len(errors) == 0, errors

    async def _extract_items_stage(
        self,
        topic: Topic,
        scenes: List[Scene],
    ) -> MemoryItemExtractResult:
        logger.info(
            f"[Stage1] Starting item extraction - Topic: {topic.topic_id}, "
            f"Scenes: {len(scenes)}"
        )

        topic_id, topic_title, topic_summary = self._format_topic_context(topic)
        scenes_content = self._format_scenes_content(scenes)
        reference_time = self._get_reference_time(scenes)

        simple_to_real = {f"scene_{i+1}": sc.scene_id for i, sc in enumerate(scenes)}
        valid_scene_ids = set(simple_to_real.keys())

        prompt = ITEM_EXTRACTION_PROMPT.format(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            scenes_content=scenes_content,
            reference_time=reference_time,
        )

        print("\n" + "=" * 80)
        print("[Stage 1] LLM Input - MemoryItem Extraction")
        print("=" * 80)
        max_display_length = 1500
        if len(prompt) > max_display_length:
            print(prompt[:max_display_length])
            print(f"\n... (truncated, total length: {len(prompt)} characters)")
        else:
            print(prompt)
        print("=" * 80)

        attempt = 0
        last_feedback = None

        while attempt < 2:
            try:
                attempt += 1

                current_prompt = (
                    prompt + f"\n\n[IMPORTANT] Previous attempt failed with the following errors, please fix:\n{last_feedback}"
                    if last_feedback
                    else prompt
                )

                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"},
                    call_site="stage2.item",
                )

                print("\n" + "=" * 80)
                print(f"[Stage 1] LLM Output (attempt {attempt})")
                print("=" * 80)
                print(resp)
                print("=" * 80)

                try:
                    data = json.loads(resp)
                except json.JSONDecodeError:
                    if HAS_JSON_REPAIR:
                        data = json_repair.loads(resp)
                    else:
                        raise

                is_valid, validation_errors = self._validate_item_extraction(
                    data, valid_scene_ids
                )

                if not is_valid:
                    raise ValueError(
                        "MemoryItem extraction validation errors:\n"
                        + "\n".join(validation_errors)
                    )

                items: List[MemoryItem] = []
                for entry in data.get("items", []):
                    item_id_val = f"item_{str(uuid.uuid4())}"

                    spatial = entry.get("spatial")
                    if isinstance(spatial, list):
                        spatial = ', '.join(str(s) for s in spatial) if spatial else None

                    temporal = entry.get("temporal")
                    if isinstance(temporal, list):
                        temporal = ', '.join(str(t) for t in temporal) if temporal else None

                    raw_scene_ids = entry.get("scene_ids", [])
                    real_scene_ids = [simple_to_real.get(sid, sid) for sid in raw_scene_ids]

                    item_obj = MemoryItem(
                        item_id=item_id_val,
                        content=entry.get("content", ""),
                        scene_ids=real_scene_ids,
                        topic_id=topic_id,
                        temporal=temporal,
                        spatial=spatial,
                        keywords=entry.get("keywords", []),
                        query_patterns=entry.get("query_patterns", []),
                        timestamp=datetime.now(),
                    )
                    items.append(item_obj)

                reasoning = data.get("reasoning", "")

                logger.info(f"[Stage1] Extracted {len(items)} items (attempt {attempt})")
                print(f"  OK: MemoryItem extraction validation passed (attempt {attempt})")

                return MemoryItemExtractResult(
                    topic_id=topic_id,
                    items=items,
                    reasoning=reasoning,
                )

            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                if isinstance(e, json.JSONDecodeError):
                    last_feedback = "JSON parsing failed. Please provide valid JSON format."
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + f"\n\nAvailable scene IDs: {', '.join(valid_scene_ids)}"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."

    async def extract_items(
        self,
        topic: Topic,
        scenes: List[Scene],
    ) -> MemoryItemExtractResult:
        """Extract memory items for a single topic; items carry scene_ids."""
        logger.info(f"[MemoryItemExtractor] Starting item extraction - Topic: {topic.topic_id}")

        item_result = await self._extract_items_stage(topic, scenes)

        if not item_result.items:
            logger.warning("[MemoryItemExtractor] No items were extracted")
            return item_result

        logger.info(
            f"[MemoryItemExtractor] Extraction complete - Items: {item_result.item_count}"
        )
        return item_result
