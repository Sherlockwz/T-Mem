"""Two-layer graph structure: L1 MemoryItemNode / L2 SceneNode / L3 TopicNode."""

import numpy as np
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

    def to_item(self) -> MemoryItem:
        return MemoryItem(
            item_id=self.id,
            content=self.content,
            scene_ids=self.scene_ids,
            topic_id=self.topic_id,
            temporal=self.temporal,
            spatial=self.spatial,
            keywords=self.keywords,
            query_patterns=self.query_patterns,
            timestamp=self.timestamp
        )

    def to_text(self) -> str:
        parts = [self.content]
        if self.temporal:
            parts.append(f"Time: {self.temporal}")
        if self.spatial:
            parts.append(f"Location: {self.spatial}")
        if len(parts) > 1:
            return f"{parts[0]} ({'; '.join(parts[1:])})"
        return self.content


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

    def to_scene(self) -> Scene:
        return Scene(
            scene_id=self.id,
            user_id_list=self.user_id_list,
            original_data=self.original_data,
            timestamp=self.timestamp if self.timestamp else datetime.now(),
            summary=self.summary if self.summary else "",
            participants=self.participants,
            type=self.type,
            keywords=self.keywords,
            subject=self.subject,
            scene_description=self.scene_description
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

    def add_node(self, layer: str, node_id: str, **kwargs):
        if layer == "item":
            self.items[node_id] = MemoryItemNode(
                id=node_id,
                content=kwargs.get("content", ""),
                scene_ids=kwargs.get("scene_ids", []),
                topic_id=kwargs.get("topic_id", ""),
                temporal=kwargs.get("temporal"),
                spatial=kwargs.get("spatial"),
                keywords=kwargs.get("keywords", []),
                query_patterns=kwargs.get("query_patterns", []),
                timestamp=kwargs.get("timestamp"),
            )

        elif layer == "scene":
            self.scenes[node_id] = SceneNode(
                id=node_id,
                user_id_list=kwargs.get("user_id_list", []),
                original_data=kwargs.get("original_data", []),
                timestamp=kwargs.get("timestamp", None),
                summary=kwargs.get("summary", ""),
                participants=kwargs.get("participants", None),
                type=kwargs.get("type", None),
                keywords=kwargs.get("keywords", None),
                subject=kwargs.get("subject", None),
                scene_description=kwargs.get("scene_description", None),
                item_ids=kwargs.get("item_ids", []),
            )

        elif layer == "topic":
            self.topics[node_id] = TopicNode(
                id=node_id,
                summary=kwargs.get("summary", ""),
                scene_ids=kwargs.get("scene_ids", []),
                timestamp=kwargs.get("timestamp", None),
                user_id_list=kwargs.get("user_id_list", []),
                participants=kwargs.get("participants", None),
            )

        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'item', 'scene', or 'topic'")

    def get_node(self, layer: str, node_id: str) -> Dict[str, Any]:
        if layer == "item":
            node = self.items.get(node_id)
            return node.model_dump() if node else {}
        elif layer == "scene":
            node = self.scenes.get(node_id)
            return node.model_dump() if node else {}
        elif layer == "topic":
            node = self.topics.get(node_id)
            return node.model_dump() if node else {}
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'item', 'scene', or 'topic'")


class MemoryGraphEmbedding(BaseModel):
    """Three-layer node embedding container."""
    model_config = {"arbitrary_types_allowed": True}

    items: Dict[str, np.ndarray] = Field(default_factory=dict)
    scenes: Dict[str, np.ndarray] = Field(default_factory=dict)
    topics: Dict[str, np.ndarray] = Field(default_factory=dict)

    def get_stats(self) -> Dict[str, int]:
        return {
            'items': len(self.items),
            'scenes': len(self.scenes),
            'topics': len(self.topics)
        }

    def to_dict(self) -> Dict[str, Any]:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        return convert_numpy({
            'items': self.items,
            'scenes': self.scenes,
            'topics': self.topics
        })

    def add_embedding(self, layer: str, node_id: str, embedding: np.ndarray):
        if layer == "item":
            self.items[node_id] = embedding
        elif layer == "scene":
            self.scenes[node_id] = embedding
        elif layer == "topic":
            self.topics[node_id] = embedding
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'item', 'scene', or 'topic'")

    def get_embedding(self, layer: str, node_id: str) -> np.ndarray:
        if layer == "item":
            return self.items.get(node_id, np.array([]))
        elif layer == "scene":
            return self.scenes.get(node_id, np.array([]))
        elif layer == "topic":
            return self.topics.get(node_id, np.array([]))
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'item', 'scene', or 'topic'")
