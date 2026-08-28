"""Topic extractor: match a scene against existing topics, else create a new one.
Topic.summary is a BM25-weighted fold of title (x2) + keywords (x1)."""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

from T_mem.utils.logger import get_logger
from T_mem.llm.llm_provider import LLMProvider
from T_mem.types import Scene, Topic
from T_mem.prompts.topic_prompts import (
    TOPIC_EXTRACTION_PROMPT,
    TOPIC_UPDATE_PROMPT,
    TOPIC_MATCH_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class SimilarScene:
    scene_id: str
    similarity_score: float
    reasoning: str


@dataclass
class SimilarTopic:
    topic_id: str
    similarity_score: float
    reasoning: str


@dataclass
class SimilarSceneResult:
    has_similar: bool
    similar_scenes: List[SimilarScene]
    reasoning: str


@dataclass
class SimilarTopicResult:
    has_similar: bool
    similar_topics: List[SimilarTopic]
    reasoning: str


@dataclass
class TopicExtractRequest:
    history_scene_list: List[Scene]
    new_scene: Scene
    existing_topics: Optional[List[Topic]] = None


@dataclass
class TopicExtractResult:
    topics: List[Topic]
    action: str  # "create_new" | "update_existing" | "merged" | "cue_only"

    similar_scene_result: Optional[SimilarSceneResult] = None
    similar_topic_result: Optional[SimilarTopicResult] = None


def _build_weighted_topic_summary(
    title: str,
    keywords: Optional[List[str]],
) -> str:
    """Fold title*2 + keywords*1 into Topic.summary; field name kept for backward compat."""
    parts: List[str] = []

    title = (title or "").strip()
    kw_list = [k for k in (keywords or []) if k]

    if title:
        parts.extend([title] * 2)

    if kw_list:
        kw_text = " ".join(kw_list)
        parts.append(kw_text)

    return " ".join(parts).strip()


class TopicExtractor:
    """LLM-driven topic matching / creation / update."""

    def __init__(
        self,
        llm_provider=LLMProvider,
        topic_match_batch_size: int = 10,
    ):
        self.llm_provider = llm_provider
        self.topic_match_batch_size = topic_match_batch_size

    def _format_scene_display(self, scene: Scene, simple_id: str = None) -> str:
        lines = [f"Scene ID: {simple_id if simple_id else scene.scene_id}"]

        if scene.participants:
            lines.append(f"Participants: {', '.join(scene.participants)}")

        if scene.subject:
            lines.append(f"Content: {scene.subject}")
        elif scene.summary:
            lines.append(f"Content: {scene.summary}")

        if scene.timestamp:
            lines.append(f"Timestamp: {scene.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if scene.keywords:
            lines.append(f"Keywords: {', '.join(scene.keywords[:8])}")

        if scene.scene_description:
            lines.append(f"Scene: {scene.scene_description}")

        return "\n".join(lines)

    def _format_scene_list(self, scene_list: List[Scene], use_simple_ids: bool = False) -> str:
        return "\n\n".join([
            f"--- Scene {i+1} ---\n{self._format_scene_display(mc, simple_id=f'scene_{i+1}' if use_simple_ids else None)}"
            for i, mc in enumerate(scene_list)
        ])

    def _format_topic_display(self, topic: Topic) -> str:
        # NOTE: topic.summary is the BM25-weighted blob (title*2+keywords*1), NOT
        # narrative text — do not surface it to the update LLM.
        lines = [
            f"Topic ID: {topic.topic_id}",
            f"Title: {topic.title}",
        ]

        if topic.timestamp:
            lines.append(f"Last Updated: {topic.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if topic.participants:
            lines.append(f"Participants: {', '.join(topic.participants)}")

        scene_count = len(topic.scene_ids)
        lines.append(f"Scene Count: {scene_count}")
        if scene_count > 0:
            if scene_count <= 5:
                lines.append(f"Scene IDs: {', '.join(topic.scene_ids)}")
            else:
                first_five = ', '.join(topic.scene_ids[:5])
                lines.append(f"Scene IDs (first 5): {first_five}, ... (+{scene_count - 5} more)")

        if topic.keywords:
            lines.append(f"Keywords: {', '.join(topic.keywords)}")

        return "\n".join(lines)

    def _validate_new_topic_extraction(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors: List[str] = []
        if "title" not in data or not data.get("title"):
            errors.append("Field 'title' missing or empty")
        if "keywords" in data and not isinstance(data.get("keywords"), list):
            errors.append("Field 'keywords' must be a list")
        return len(errors) == 0, errors

    def _validate_topic_update(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        return self._validate_new_topic_extraction(data)

    async def _extract_new_topic(self, scene_list: List[Scene]) -> Optional[Topic]:
        scenes_text = self._format_scene_list(scene_list)
        logger.info(f"[Stage3] Creating new topic - scene count: {len(scene_list)}")

        prompt = TOPIC_EXTRACTION_PROMPT.format(
            scenes=scenes_text,
        )

        print("\n" + "=" * 80)
        print("[Stage 3] LLM Input - Create New Topic")
        print("=" * 80)
        max_display_length = 1000
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
                    call_site="stage2.topic.new",
                )

                print("\n" + "=" * 80)
                print(f"[Stage 3] LLM Output (attempt {attempt})")
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

                is_valid, validation_errors = self._validate_new_topic_extraction(data)
                if not is_valid:
                    raise ValueError(
                        "New topic extraction validation errors:\n"
                        + "\n".join(validation_errors)
                    )

                import uuid
                topic_id_val = f"topic_{str(uuid.uuid4())}"

                scene_ids = [mc.scene_id for mc in scene_list]

                user_id_set: set = set()
                participant_set: set = set()
                for mc in scene_list:
                    user_id_set.update(mc.user_id_list)
                    if mc.participants:
                        participant_set.update(mc.participants)

                last_scene = scene_list[-1]
                timestamp = last_scene.timestamp
                if isinstance(timestamp, int):
                    timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                raw_title = data.get("title", "") or ""
                raw_keywords = data.get("keywords", []) or []

                weighted_summary = _build_weighted_topic_summary(
                    title=raw_title,
                    keywords=raw_keywords,
                )

                topic = Topic(
                    topic_id=topic_id_val,
                    title=raw_title,
                    summary=weighted_summary,
                    scene_ids=scene_ids,
                    timestamp=timestamp,
                    user_id_list=list(user_id_set),
                    participants=list(participant_set) if participant_set else None,
                    keywords=raw_keywords,
                )

                logger.info(f"[Stage3] Created new topic: {topic.title} (attempt {attempt})")
                print(f"  ✓ New topic extraction validation passed (attempt {attempt})")

                return topic

            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                if isinstance(e, json.JSONDecodeError):
                    last_feedback = (
                        "JSON parsing failed. Please provide valid JSON format with:"
                        "\n{{\n  \"title\": \"...\",\n  \"keywords\": [...]\n}}"
                    )
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = (
                        str(e)
                        + "\n\nPlease ensure:\n1. 'title' is present and not empty\n"
                        + "2. 'keywords' is optional but must be a list if provided"
                    )
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."

    # -------- Stage 3: update --------

    async def _update_existing_topic(
        self,
        topic: Topic,
        new_scene: Scene,
    ) -> Optional[Topic]:
        topic_text = self._format_topic_display(topic)
        scene_text = self._format_scene_display(new_scene)
        logger.info(f"[Stage3] Updating topic: {topic.topic_id}")

        prompt = TOPIC_UPDATE_PROMPT.format(
            existing_topic=topic_text,
            new_scene=scene_text,
        )

        print("\n" + "=" * 80)
        print(f"[Stage 3] LLM Input - Update Topic {topic.topic_id}")
        print("=" * 80)
        max_display_length = 1000
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
                    call_site="stage2.topic.update",
                )

                print("\n" + "=" * 80)
                print(f"[Stage 3] LLM Output - Topic {topic.topic_id} (attempt {attempt})")
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

                is_valid, validation_errors = self._validate_topic_update(data)
                if not is_valid:
                    raise ValueError(
                        "Topic update validation errors:\n"
                        + "\n".join(validation_errors)
                    )

                updated_scene_ids = list(topic.scene_ids)
                if new_scene.scene_id not in updated_scene_ids:
                    updated_scene_ids.append(new_scene.scene_id)

                user_id_set = set(topic.user_id_list)
                user_id_set.update(new_scene.user_id_list)

                participant_set = set(topic.participants) if topic.participants else set()
                if new_scene.participants:
                    participant_set.update(new_scene.participants)

                timestamp = new_scene.timestamp
                if isinstance(timestamp, int):
                    timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                raw_title = data.get("title", topic.title) or topic.title or ""
                raw_keywords = data.get("keywords", topic.keywords) or topic.keywords or []

                weighted_summary = _build_weighted_topic_summary(
                    title=raw_title,
                    keywords=raw_keywords,
                )

                updated_topic = Topic(
                    topic_id=topic.topic_id,
                    title=raw_title,
                    summary=weighted_summary,
                    scene_ids=updated_scene_ids,
                    timestamp=timestamp,
                    user_id_list=list(user_id_set),
                    participants=list(participant_set) if participant_set else None,
                    keywords=raw_keywords,
                )

                logger.info(f"[Stage3] Updated topic: {updated_topic.title} (attempt {attempt})")
                print(f"  ✓ Topic update validation passed (attempt {attempt})")

                return updated_topic

            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                if isinstance(e, json.JSONDecodeError):
                    last_feedback = (
                        "JSON parsing failed. Please provide valid JSON format with:"
                        "\n{{\n  \"title\": \"...\",\n  \"keywords\": [...]\n}}"
                    )
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = (
                        str(e)
                        + "\n\nPlease ensure:\n1. 'title' is present and not empty\n"
                        + "2. 'keywords' is optional but must be a list if provided"
                    )
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."

    # -------- Stage 2: topic matching --------

    async def _llm_match_topics_batch(
        self,
        scene: Scene,
        topics_batch: List[Topic],
        batch_id_offset: int = 0,
    ) -> List[str]:
        sc_subject = scene.subject or ""
        sc_summary = scene.summary or ""

        topic_lines = []
        simple_to_real: Dict[str, str] = {}
        for i, t in enumerate(topics_batch):
            simple_id = f"topic_{batch_id_offset + i + 1}"
            simple_to_real[simple_id] = t.topic_id
            # NOTE: t.summary is the BM25-weighted blob; use t.keywords instead
            # for semantic matching signal.
            kw_text = ", ".join(t.keywords) if t.keywords else ""
            topic_lines.append(f"- {simple_id}: {t.title}\n  Keywords: {kw_text}")
        topics_text = "\n".join(topic_lines)

        prompt = TOPIC_MATCH_PROMPT.format(
            scene_subject=sc_subject,
            scene_summary=sc_summary,
            num_topics=len(topics_batch),
            topics_text=topics_text,
        )

        resp = await self.llm_provider.generate(prompt, response_format={"type": "json_object"},
                                                call_site="stage2.topic.match")

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            if HAS_JSON_REPAIR:
                data = json_repair.loads(resp)
            else:
                raise

        matched_ids: List[str] = []
        for item in data.get("results", []):
            if item.get("match") is True:
                simple_id = item.get("topic_id", "")
                real_id = simple_to_real.get(simple_id)
                if real_id:
                    matched_ids.append(real_id)
        return matched_ids

    async def _llm_match_topics(
        self,
        scene: Scene,
        existing_topics: List[Topic],
    ) -> List[str]:
        batch_size = self.topic_match_batch_size
        all_matched_ids: List[str] = []

        for i in range(0, len(existing_topics), batch_size):
            batch = existing_topics[i:i + batch_size]
            matched = await self._llm_match_topics_batch(scene, batch, batch_id_offset=i)
            all_matched_ids.extend(matched)

        return all_matched_ids

    # -------- Public entry --------

    async def extract_topic(
        self,
        request: TopicExtractRequest,
    ) -> Optional[TopicExtractResult]:
        """Topic extraction pipeline: match → create or update."""
        logger.info("[TopicExtractor] Starting topic extraction")

        if not request.existing_topics:
            logger.info("No existing topics, creating a new topic")
            topic = await self._extract_new_topic([request.new_scene])
            if topic:
                return TopicExtractResult(topics=[topic], action="create_new")
            return None

        matched_topic_ids = await self._llm_match_topics(
            scene=request.new_scene,
            existing_topics=request.existing_topics,
        )
        logger.info(f"LLM matched {len(matched_topic_ids)} topics: {matched_topic_ids}")
        print(
            f"  [Topic Match] Scene matched {len(matched_topic_ids)}/"
            f"{len(request.existing_topics)} topics"
        )

        if not matched_topic_ids:
            logger.info("No matching topics, creating a new topic")
            topic = await self._extract_new_topic([request.new_scene])
            if topic:
                return TopicExtractResult(topics=[topic], action="create_new")
            return None

        matched_topics = [t for t in request.existing_topics if t.topic_id in matched_topic_ids]
        logger.info(f"Updating {len(matched_topics)} matched topics")

        updated_topics: List[Topic] = []
        for topic_to_update in matched_topics:
            updated_topic = await self._update_existing_topic(
                topic=topic_to_update,
                new_scene=request.new_scene,
            )
            if updated_topic:
                updated_topics.append(updated_topic)
                logger.info(f"  Updated topic: {updated_topic.title}")

        if updated_topics:
            return TopicExtractResult(topics=updated_topics, action="update_existing")

        return None
