"""Tests for SharedContext."""

from agenticflow.models import SharedContext


class TestSharedContext:
    def test_set_and_get(self):
        ctx = SharedContext()
        ctx.set("key1", "value1")
        assert ctx.get("key1") == "value1"

    def test_get_default(self):
        ctx = SharedContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_delete(self):
        ctx = SharedContext()
        ctx.set("key1", "value1")
        assert ctx.delete("key1") is True
        assert ctx.get("key1") is None
        assert ctx.delete("key1") is False

    def test_keys(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        assert sorted(ctx.keys()) == ["a", "b"]

    def test_len(self):
        ctx = SharedContext()
        assert len(ctx) == 0
        ctx.set("a", 1)
        assert len(ctx) == 1

    def test_contains(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        assert "a" in ctx
        assert "b" not in ctx

    def test_to_prompt_fragment_empty(self):
        ctx = SharedContext()
        assert ctx.to_prompt_fragment() == ""

    def test_to_prompt_fragment_with_data(self):
        ctx = SharedContext()
        ctx.set("plan", "Build a REST API")
        ctx.set("language", "Python")
        fragment = ctx.to_prompt_fragment()
        assert "Shared context" in fragment
        assert "plan" in fragment
        assert "Build a REST API" in fragment
        assert "language" in fragment

    def test_to_prompt_fragment_truncates_long_values(self):
        ctx = SharedContext()
        ctx.set("long", "x" * 5000)
        fragment = ctx.to_prompt_fragment()
        assert "truncated" in fragment
        assert len(fragment) < 5000

    def test_overwrite(self):
        ctx = SharedContext()
        ctx.set("key", "old")
        ctx.set("key", "new")
        assert ctx.get("key") == "new"
        assert len(ctx) == 1
