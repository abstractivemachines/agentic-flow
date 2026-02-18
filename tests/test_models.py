"""Tests for agenticflow.models."""

from pathlib import Path

import pytest

from agenticflow.models import AgentType, Task, TaskResult, TaskStatus, Workspace


class TestTask:
    def test_defaults(self):
        t = Task(description="Do something")
        assert t.status == TaskStatus.PENDING
        assert t.agent_type is None
        assert t.context == {}
        assert len(t.id) == 8

    def test_to_prompt(self):
        t = Task(description="Build a widget", context={"lang": "python"})
        prompt = t.to_prompt()
        assert "Build a widget" in prompt
        assert "python" in prompt

    def test_to_prompt_no_context(self):
        t = Task(description="Simple task")
        prompt = t.to_prompt()
        assert prompt == "Task: Simple task"


class TestTaskResult:
    def test_success(self):
        r = TaskResult(
            task_id="abc",
            agent_type=AgentType.CODER,
            success=True,
            summary="Done",
        )
        assert r.success
        assert r.errors == []
        assert r.artifacts == {}

    def test_failure(self):
        r = TaskResult(
            task_id="abc",
            agent_type=AgentType.TESTER,
            success=False,
            summary="Tests failed",
            errors=["assertion error"],
        )
        assert not r.success
        assert len(r.errors) == 1


class TestWorkspace:
    def test_creates_directory(self, tmp_path: Path):
        ws_path = tmp_path / "new_workspace"
        ws = Workspace(root=ws_path)
        assert ws.root.exists()

    def test_resolve_valid(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        resolved = ws.resolve("subdir/file.py")
        assert str(resolved).startswith(str(tmp_path.resolve()))

    def test_resolve_traversal(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        try:
            ws.resolve("../../etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_resolve_sibling_prefix_traversal(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        with pytest.raises(ValueError):
            ws.resolve("../ws2/secrets.txt")

    def test_results_tracking(self, tmp_path: Path):
        ws = Workspace(root=tmp_path)
        assert ws.get_results_summary() == "No completed tasks yet."

        ws.add_result(TaskResult(
            task_id="1", agent_type=AgentType.CODER,
            success=True, summary="Built it",
        ))
        summary = ws.get_results_summary()
        assert "SUCCESS" in summary
        assert "coder" in summary
