"""Shared pytest fixtures."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

# Make the repo importable without installation (also supports `pip install -e .`).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_scene():
    from T_mem.types import Scene

    return Scene(
        scene_id="scene_1",
        user_id_list=["u_1", "u_2"],
        original_data=[{"speaker": "u_1", "content": "hello"}],
        timestamp=datetime.datetime(2024, 3, 14, tzinfo=datetime.timezone.utc),
        summary="A short greeting.",
    )


@pytest.fixture
def sample_item():
    from T_mem.types import MemoryItem

    return MemoryItem(
        item_id="item_1",
        content="Alice likes hiking on weekends.",
        scene_ids=["scene_1"],
        topic_id="topic_1",
        temporal="weekends",
        keywords=["hiking", "Alice"],
    )


@pytest.fixture
def sample_topic():
    from T_mem.types import Topic

    return Topic(
        topic_id="topic_1",
        title="Hiking",
        summary="Alice and Bob often discuss hiking.",
        scene_ids=["scene_1", "scene_2"],
        timestamp=datetime.datetime(2024, 3, 14, tzinfo=datetime.timezone.utc),
        user_id_list=["u_1", "u_2"],
    )
