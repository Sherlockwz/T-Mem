"""Two-layer graph structure: L1 MemoryItemNode / L2 SceneNode / L3 TopicNode."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .types import MemoryItem, Scene, Topic, RawDataType


class MemoryItemNode(BaseModel):
    """L1 layer MemoryItem node."""
    id: str = Field(..., description="MemoryItem unique identifier (item_id)")
    content: str = Field(..., description="MemoryItem content")
    scene_ids: List[str] = Field(default_factory=list, description="List of scene IDs that compose this item")
    topic_id: str = Field(default="", description="Source topic ID")

    temporal: Optional[str] = Field(default=None, description="Time information, format: 'relative time (absolute time)'")
    spatial: Optional[str] = Field(default=None, description="Location information")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    query_patterns: List[str] = Field(default_factory=list, description="Query patterns that can be answered")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp")

    @classmethod
    def from_item(cls, item: MemoryItem) -> 'MemoryItemNode':
        return cls(
            id=item.item_id,
            content=item.content,
            scene_ids=item.scene_ids or [],
            topic_id=item.topic_id or "",
            temporal=item.temporal,
            spatial=item.spatial,
            keywords=item.keywords or [],
            query_patterns=item.query_patterns or [],
            timestamp=item.timestamp,
        )


class SceneNode(BaseModel):
    """L2 Scene node holding item_ids list."""
    id: str = Field(..., description="Node unique identifier, corresponds to scene_id")

    user_id_list: List[str] = Field(default_factory=list, description="Involved user ID list")
    original_data: List[Dict[str, Any]] = Field(default_factory=list, description="Original data")
    timestamp: Optional[datetime] = Field(default=None, description="Scene timestamp")
    summary: Optional[str] = Field(default=None, description="Scene summary")

    participants: Optional[List[str]] = Field(default=None, description="Participant list")
    type: Optional[RawDataType] = Field(default=None, description="Raw data type")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords extracted from scene")
    subject: Optional[str] = Field(default=None, description="Scene subject")
    scene_description: Optional[str] = Field(default=None, description="Scene memory description")

    item_ids: List[str] = Field(
        default_factory=list,
        description="List of item IDs that belong to this scene"
    )

    @classmethod
    def from_scene(cls, scene: Scene, item_ids: Optional[List[str]] = None) -> 'SceneNode':
        return cls(
            id=scene.scene_id,
            user_id_list=scene.user_id_list,
            original_data=scene.original_data,
            timestamp=scene.timestamp,
            summary=scene.summary,
            participants=scene.participants,
            type=scene.type,
            keywords=scene.keywords,
            subject=scene.subject,
            scene_description=scene.scene_description,
            item_ids=item_ids or [],
        )


class TopicNode(BaseModel):
    """L3 Topic node; summary is BM25/embedding text (title*2 + keywords*1)."""
    id: str = Field(..., description="Node unique identifier, corresponds to topic_id")

    summary: str = Field(..., description="Pre-weighted BM25/embedding text (title*2 + keywords*1)")

    scene_ids: List[str] = Field(
        default_factory=list,
        description="List of scene IDs that belong to this topic"
    )

    timestamp: Optional[datetime] = Field(default=None, description="Topic creation time (time of last scene)")
    user_id_list: List[str] = Field(default_factory=list, description="Involved user ID list")
    participants: Optional[List[str]] = Field(default=None, description="Participant list")

    @classmethod
    def from_topic(cls, topic: Topic) -> 'TopicNode':
        return cls(
            id=topic.topic_id,
            summary=topic.summary,
            scene_ids=topic.scene_ids,
            timestamp=topic.timestamp,
            user_id_list=topic.user_id_list,
            participants=topic.participants,
        )


class MemoryGraph(BaseModel):
    """Three-layer graph: L1 items, L2 scenes (with item_ids), L3 topics (with scene_ids)."""
    items: Dict[str, MemoryItemNode] = Field(
        default_factory=dict,
        description="MemoryItem node dictionary"
    )

    scenes: Dict[str, SceneNode] = Field(
        default_factory=dict,
        description="Scene node dictionary, keyed by scene ID"
    )

    topics: Dict[str, TopicNode] = Field(
        default_factory=dict,
        description="Topic node dictionary, keyed by topic ID"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'items': {k: v.model_dump(mode='json') for k, v in self.items.items()},
            'scenes': {k: v.model_dump(mode='json') for k, v in self.scenes.items()},
            'topics': {k: v.model_dump(mode='json') for k, v in self.topics.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryGraph':
        return cls(
            items={k: MemoryItemNode(**v) for k, v in data.get('items', {}).items()},
            scenes={k: SceneNode(**v) for k, v in data.get('scenes', {}).items()},
            topics={k: TopicNode(**v) for k, v in data.get('topics', {}).items()}
        )

    def get_stats(self) -> Dict[str, int]:
        return {
            'items': len(self.items),
            'scenes': len(self.scenes),
            'topics': len(self.topics)
        }
