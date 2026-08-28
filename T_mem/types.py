"""Core data types: Scene, MemoryItem, Topic."""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import datetime
import logging

from T_mem.utils.datetime_utils import to_iso_format

logger = logging.getLogger(__name__)


class RawDataType(Enum):
    CONVERSATION = "Conversation"


@dataclass
class Scene:
    """Scene representing a conversation segment."""
    scene_id: str
    user_id_list: List[str]
    original_data: List[Dict[str, Any]]
    timestamp: datetime.datetime
    summary: str

    participants: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    keywords: Optional[List[str]] = None
    subject: Optional[str] = None
    scene_description: Optional[str] = None

    def __post_init__(self):
        if not self.scene_id:
            raise ValueError("scene_id is required")
        if not self.original_data:
            raise ValueError("original_data is required")
        if not self.summary:
            raise ValueError("summary is required")

    def __repr__(self) -> str:
        return f"Scene(scene_id={self.scene_id}, original_data={self.original_data}, timestamp={self.timestamp}, summary={self.summary})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),
            "summary": self.summary,
            "participants": self.participants,
            "type": str(self.type.value) if self.type else None,
            "keywords": self.keywords,
            "subject": self.subject,
            "scene_description": self.scene_description,
        }


@dataclass
class MemoryItem:
    """Semantically complete information unit extracted from topics."""
    item_id: str
    content: str
    scene_ids: List[str]
    topic_id: str

    temporal: Optional[str] = None
    spatial: Optional[str] = None
    keywords: Optional[List[str]] = None
    query_patterns: Optional[List[str]] = None
    timestamp: Optional[datetime.datetime] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.query_patterns is None:
            self.query_patterns = []

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"MemoryItem(id={self.item_id}, content='{content_preview}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "scene_ids": self.scene_ids,
            "topic_id": self.topic_id,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "keywords": self.keywords,
            "query_patterns": self.query_patterns,
            "timestamp": to_iso_format(self.timestamp) if self.timestamp else None
        }


@dataclass
class Topic:
    """Abstraction over related scenes (recurring pattern / context / activity)."""
    topic_id: str
    title: str
    summary: str
    scene_ids: List[str]
    timestamp: datetime.datetime
    user_id_list: List[str]

    participants: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

    def __post_init__(self):
        if not self.topic_id:
            raise ValueError("topic_id is required")
        if not self.title:
            raise ValueError("title is required")
        if not self.summary:
            raise ValueError("summary is required")
        if not self.scene_ids:
            raise ValueError("scene_ids is required")

    def __repr__(self) -> str:
        return f"Topic(topic_id={self.topic_id}, title={self.title}, scene_count={len(self.scene_ids)})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "summary": self.summary,
            "scene_ids": self.scene_ids,
            "timestamp": to_iso_format(self.timestamp),
            "user_id_list": self.user_id_list,
            "participants": self.participants,
            "keywords": self.keywords,
        }
