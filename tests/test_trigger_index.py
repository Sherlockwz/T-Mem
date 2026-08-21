"""Tests for T_mem.index.trigger_index (EntityBridgeTrigger)."""

from __future__ import annotations

import json

from T_mem.index.trigger_index import EntityBridgeTrigger


class TestEntityBridgeTrigger:
    def test_defaults(self):
        t = EntityBridgeTrigger(concept="hiking", concept_norm="hiking")
        assert t.id.startswith("eb_")
        assert t.item_confidences == {}
        assert t.quality == 0.0
        assert t.item_ids == []

    def test_add_item_keeps_max(self):
        t = EntityBridgeTrigger(concept="c", concept_norm="c")
        t.add_item("i1", 0.5)
        t.add_item("i1", 0.9)
        assert t.item_confidences["i1"] == 0.9
        assert t.quality == 0.9

    def test_filter_items_threshold(self):
        t = EntityBridgeTrigger(concept="c", concept_norm="c")
        t.add_item("i1", 0.9)
        t.add_item("i2", 0.5)
        t.filter_items(0.7)
        assert list(t.item_confidences) == ["i1"]

    def test_merge_from(self):
        a = EntityBridgeTrigger(concept="c", concept_norm="c")
        a.add_item("i1", 0.8)
        a.activation_patterns = ["pat1"]
        b = EntityBridgeTrigger(concept="c", concept_norm="c")
        b.add_item("i2", 0.6)
        b.activation_patterns = ["pat1", "pat2"]
        b.bridge = "longer bridge text"
        a.merge_from(b)
        assert set(a.item_confidences) == {"i1", "i2"}
        assert a.activation_patterns == ["pat1", "pat2"]
        assert a.bridge == "longer bridge text"

    def test_to_from_dict_roundtrip(self):
        t = EntityBridgeTrigger(
            concept="Hiking",
            concept_norm="hiking",
            bridge="outdoor",
            activation_patterns=["hike"],
            item_confidences={"i1": 0.85},
        )
        d = t.to_dict()
        assert d["quality"] == 0.85
        restored = EntityBridgeTrigger.from_dict(d)
        assert restored.concept == "Hiking"
        assert restored.item_confidences == {"i1": 0.85}

    def test_to_dict_json_serializable(self):
        t = EntityBridgeTrigger(concept="c", concept_norm="c")
        json.dumps(t.to_dict())  # must not raise

    def test_to_text_embedding_modes(self, monkeypatch):
        t = EntityBridgeTrigger(
            concept="hiking",
            concept_norm="hiking",
            bridge="weekend activity",
            activation_patterns=["trail"],
        )
        monkeypatch.setenv("TRIGGER_EMBED_TEXT_MODE", "concept_only")
        text = t.to_text_for_embedding()
        assert "hiking" in text
        monkeypatch.setenv("TRIGGER_EMBED_TEXT_MODE", "concept_bridge")
        text = t.to_text_for_embedding()
        assert "weekend activity" in text
