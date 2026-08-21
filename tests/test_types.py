"""Tests for T_mem.types data classes."""

from __future__ import annotations

import datetime

import pytest

from T_mem.types import MemoryItem, RawDataType, Scene, Topic


class TestScene:
    def test_valid(self, sample_scene):
        assert sample_scene.scene_id == "scene_1"
        assert sample_scene.to_dict()["timestamp"] is not None
        assert sample_scene.to_dict()["type"] is None

    def test_missing_scene_id(self):
        with pytest.raises(ValueError):
            Scene(
                scene_id="",
                user_id_list=[],
                original_data=[{"speaker": "a"}],
                timestamp=datetime.datetime.now(),
                summary="s",
            )

    def test_missing_summary(self):
        with pytest.raises(ValueError):
            Scene(
                scene_id="s1",
                user_id_list=[],
                original_data=[{"speaker": "a"}],
                timestamp=datetime.datetime.now(),
                summary="",
            )

    def test_to_dict_roundtrip_type(self):
        s = Scene(
            scene_id="s1",
            user_id_list=["u1"],
            original_data=[{"speaker": "a"}],
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            summary="sum",
            type=RawDataType.CONVERSATION,
        )
        d = s.to_dict()
        assert d["type"] == "Conversation"


class TestMemoryItem:
    def test_defaults(self):
        item = MemoryItem(
            item_id="i1",
            content="content",
            scene_ids=["s1"],
            topic_id="t1",
        )
        assert item.keywords == []
        assert item.query_patterns == []

    def test_to_text_with_temporal(self):
        item = MemoryItem(
            item_id="i1",
            content="Alice likes hiking.",
            scene_ids=["s1"],
            topic_id="t1",
            temporal="weekends",
        )
        assert "weekends" in item.to_text()

    def test_to_dict_contains_keys(self, sample_item):
        d = sample_item.to_dict()
        for key in ("item_id", "content", "scene_ids", "topic_id", "temporal", "spatial"):
            assert key in d


class TestTopic:
    def test_valid(self, sample_topic):
        assert sample_topic.to_dict()["scene_ids"] == ["scene_1", "scene_2"]

    def test_missing_title(self):
        with pytest.raises(ValueError):
            Topic(
                topic_id="t1",
                title="",
                summary="s",
                scene_ids=["s1"],
                timestamp=datetime.datetime.now(),
                user_id_list=[],
            )

    def test_missing_scene_ids(self):
        with pytest.raises(ValueError):
            Topic(
                topic_id="t1",
                title="title",
                summary="s",
                scene_ids=[],
                timestamp=datetime.datetime.now(),
                user_id_list=[],
            )
