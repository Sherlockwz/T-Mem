"""Tests for T_mem.utils.cost_ledger (no-op unless T_MEM_COST_LOG is set)."""

from __future__ import annotations

import json
import os

import pytest

from T_mem.utils import cost_ledger


@pytest.fixture(autouse=True)
def _cleanup_env():
    old = os.environ.get("T_MEM_COST_LOG")
    os.environ.pop("T_MEM_COST_LOG", None)
    yield
    if old is not None:
        os.environ["T_MEM_COST_LOG"] = old
    else:
        os.environ.pop("T_MEM_COST_LOG", None)


class TestEnabled:
    def test_disabled_by_default(self):
        assert cost_ledger.enabled() is False

    def test_enabled_when_env_set(self, tmp_path):
        os.environ["T_MEM_COST_LOG"] = str(tmp_path / "ledger.jsonl")
        assert cost_ledger.enabled() is True


class TestRecord:
    def test_noop_when_disabled(self, tmp_path):
        cost_ledger.record(prompt="p", completion="c", model="m")
        assert not (tmp_path / "ledger.jsonl").exists()

    def test_writes_jsonl_row(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        os.environ["T_MEM_COST_LOG"] = str(ledger)
        cost_ledger.record(
            prompt="hello world",
            completion="hi",
            model="gpt-4o-mini",
            call_site="test.site",
            conv_id="conv_1",
            api_usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            latency_s=0.5,
        )
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["model"] == "gpt-4o-mini"
        assert row["call_site"] == "test.site"
        assert row["conv_id"] == "conv_1"
        assert row["token_source"] == "api"
        assert row["prompt_tokens"] == 5
        assert row["completion_tokens"] == 1

    def test_never_raises(self, tmp_path):
        os.environ["T_MEM_COST_LOG"] = str(tmp_path / "ledger.jsonl")
        # completion=None is not valid input; record should swallow any error.
        cost_ledger.record(prompt="p", completion=None, model="m")  # type: ignore[arg-type]


class TestConvContext:
    def test_conv_contextvar(self):
        cost_ledger.set_conv("conv_42")
        assert cost_ledger.get_conv() == "conv_42"
        cost_ledger.set_conv(None)
        assert cost_ledger.get_conv() is None
