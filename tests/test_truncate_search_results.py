"""Tests for T_mem.io.truncate_search_results.truncate_context."""

from __future__ import annotations

import re

from T_mem.io.truncate_search_results import truncate_context


def _make_context(n_scene: int, n_item: int) -> str:
    """Build a context string mirroring stage6 format_hierarchical_results."""
    parts = ["## Relevant Scenes:\n"]
    for i in range(1, n_scene + 1):
        parts.append(f"[Scene {i}] summary of scene {i}\n  Time: T{i}")
        if i < n_scene:
            parts.append("\n\n")
    parts.append("\n\n## Relevant Items:\n")
    for i in range(1, n_item + 1):
        parts.append(f"[Item {i}] content of item {i}")
        if i < n_item:
            parts.append("\n\n")
    parts.append("\n")
    return "".join(parts)


class TestTruncateContext:
    def test_keep_both(self):
        ctx = _make_context(5, 15)
        out = truncate_context(ctx, keep_scene=3, keep_item=5)
        assert len(re.findall(r"^\[Scene \d+\]", out, re.M)) == 3
        assert len(re.findall(r"^\[Item \d+\]", out, re.M)) == 5

    def test_renumbered(self):
        ctx = _make_context(4, 6)
        out = truncate_context(ctx, keep_scene=2, keep_item=3)
        assert "[Scene 1]" in out and "[Scene 2]" in out
        assert "[Scene 3]" not in out
        assert "[Item 1]" in out and "[Item 3]" in out
        assert "[Item 4]" not in out

    def test_no_scenes_section(self):
        ctx = "## Relevant Items:\n[Item 1] a\n\n[Item 2] b\n"
        out = truncate_context(ctx, keep_scene=5, keep_item=1)
        assert "[Item 2]" not in out
        assert "[Item 1]" in out

    def test_keep_more_than_available(self):
        ctx = _make_context(2, 3)
        out = truncate_context(ctx, keep_scene=10, keep_item=10)
        assert len(re.findall(r"^\[Scene \d+\]", out, re.M)) == 2
        assert len(re.findall(r"^\[Item \d+\]", out, re.M)) == 3

    def test_keep_zero(self):
        ctx = _make_context(3, 3)
        out = truncate_context(ctx, keep_scene=0, keep_item=0)
        assert "No relevant scenes found." in out
        assert "No relevant items found." in out

    def test_section_headers_preserved(self):
        ctx = _make_context(3, 3)
        out = truncate_context(ctx, keep_scene=1, keep_item=1)
        assert "## Relevant Scenes:\n" in out
        assert "## Relevant Items:\n" in out
