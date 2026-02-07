"""Tests for the orchestrator — using mocked Anthropic client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_tool_use_block(
    tool_id: str, name: str, tool_input: dict[str, Any]
) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _make_response(
    content: list[Any], stop_reason: str = "end_turn"
) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class TestOrchestrator:
    def test_single_turn(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(workspace=ws)
        orch.client = MagicMock()
        orch.client.messages.create.return_value = _make_response(
            [_make_text_block("All done!")]
        )
        result = orch.run("Do something")
        assert "All done!" in result

    def test_dispatch_agent(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(workspace=ws)
        orch.client = MagicMock()

        # First response: dispatch to coder
        dispatch_call = _make_tool_use_block(
            "t1",
            "dispatch_agent",
            {"agent_type": "coder", "task_description": "Write hello.py"},
        )
        first_response = _make_response([dispatch_call], stop_reason="tool_use")

        # Second response: done
        second_response = _make_response([_make_text_block("Finished.")])

        orch.client.messages.create.side_effect = [first_response, second_response]

        # Mock the agent dispatch so we don't need real API
        with patch.object(orch, "_dispatch", return_value="[SUCCESS] coder: done"):
            result = orch.run("Build something")
        assert "Finished." in result

    def test_agent_models_config(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(
            workspace=ws,
            model="default-model",
            agent_models={"planner": "opus-model", "coder": "sonnet-model"},
        )
        assert orch.agent_models["planner"] == "opus-model"
        assert orch.agent_models["coder"] == "sonnet-model"

    def test_set_shared_context_tool(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(workspace=ws)
        orch.client = MagicMock()

        # First response: set context
        ctx_call = _make_tool_use_block(
            "t1",
            "set_shared_context",
            {"key": "plan", "value": "Build a REST API"},
        )
        first_response = _make_response([ctx_call], stop_reason="tool_use")
        second_response = _make_response([_make_text_block("Done.")])

        orch.client.messages.create.side_effect = [first_response, second_response]
        result = orch.run("Plan something")

        assert "Done." in result
        assert ws.shared_context.get("plan") == "Build a REST API"

    def test_unknown_tool(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(workspace=ws)
        orch.client = MagicMock()

        unknown_call = _make_tool_use_block("t1", "bad_tool", {})
        first_response = _make_response([unknown_call], stop_reason="tool_use")
        second_response = _make_response([_make_text_block("OK")])

        orch.client.messages.create.side_effect = [first_response, second_response]
        result = orch.run("test")
        assert "OK" in result

    def test_max_turns(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = Orchestrator(workspace=ws)
        orch.client = MagicMock()

        # Always dispatch, never finish
        dispatch_call = _make_tool_use_block(
            "t1",
            "dispatch_agent",
            {"agent_type": "coder", "task_description": "loop"},
        )
        looping_response = _make_response([dispatch_call], stop_reason="tool_use")
        orch.client.messages.create.return_value = looping_response

        with patch.object(orch, "_dispatch", return_value="[SUCCESS] done"):
            result = orch.run("loop forever")
        assert "maximum turns" in result.lower()
