"""Tests for T_mem.llm.llm_provider (no network: JSON parsing / retry wiring / ledger hooks)."""

from __future__ import annotations

import json

import pytest

from T_mem.llm.llm_provider import (
    LLMProvider,
    _extract_usage,
    _record_cost_safe,
    JsonFailureLogger,
)


class TestJsonParsing:
    def test_plain_json(self):
        assert LLMProvider._safe_parse_json('{"a": 1}') == ({"a": 1}, None)

    def test_code_fenced_json(self):
        raw = '```json\n{"a": 1}\n```'
        obj, err = LLMProvider._safe_parse_json(raw)
        assert err is None
        assert obj == {"a": 1}

    def test_plain_fence_without_lang(self):
        raw = '```\n{"a": 1}\n```'
        obj, err = LLMProvider._safe_parse_json(raw)
        assert err is None
        assert obj == {"a": 1}

    def test_invalid_returns_error(self):
        obj, err = LLMProvider._safe_parse_json("not json at all")
        assert obj is None
        assert err is not None

    def test_empty(self):
        obj, err = LLMProvider._safe_parse_json("")
        assert obj is None
        assert err is not None


class TestUsageExtraction:
    def test_openai_style(self):
        ret = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        assert _extract_usage(ret)["prompt_tokens"] == 10

    def test_data_wrapped(self):
        ret = {"data": {"usage": {"prompt_tokens": 3}}}
        assert _extract_usage(ret)["prompt_tokens"] == 3

    def test_missing(self):
        assert _extract_usage({"choices": []}) is None
        assert _extract_usage(None) is None


class TestRecordCostSafe:
    def test_noop_when_ledger_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("T_MEM_COST_LOG", raising=False)
        # Must not raise and must not create files.
        _record_cost_safe("p", "c", "m", {"usage": {}}, {"call_site": "t"}, 0.1)

    def test_writes_row_when_enabled(self, tmp_path, monkeypatch):
        ledger = tmp_path / "ledger.jsonl"
        monkeypatch.setenv("T_MEM_COST_LOG", str(ledger))
        _record_cost_safe(
            "prompt", "completion", "gpt-4o-mini",
            {"usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            {"call_site": "stage1", "conv_id": "c1"},
            0.2,
        )
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["call_site"] == "stage1"
        assert row["conv_id"] == "c1"


class TestJsonFailureLogger:
    def test_writes_entry(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        logger = JsonFailureLogger(str(path))
        logger.log(
            conv_id="c1",
            call_site="stage2",
            attempt_count=3,
            last_error="parse failed",
            prompt_preview="prompt...",
            raw_response="bad json",
            model="gpt-4o-mini",
        )
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["call_site"] == "stage2"
        assert entry["last_error"] == "parse failed"

    def test_append_multiple(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        logger = JsonFailureLogger(str(path))
        logger.log(conv_id="c1", call_site="s")
        logger.log(conv_id="c2", call_site="s")
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
