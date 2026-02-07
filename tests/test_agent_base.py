"""Tests for agenticflow.agents.base — using a mocked Anthropic client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from agenticflow.agents.base import Agent
from agenticflow.models import AgentType, Task, Workspace


class StubAgent(Agent):
    """Concrete agent subclass for testing."""

    agent_type = AgentType.CODER
    system_prompt = "You are a test agent."


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


class TestAgentLoop:
    def test_single_turn_text_response(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        agent = StubAgent(workspace=ws)

        mock_response = _make_response([_make_text_block("All done!")])
        agent.client = MagicMock()
        agent.client.messages.create.return_value = mock_response

        task = Task(description="Do something")
        result = agent.run(task)

        assert result.success
        assert "All done!" in result.summary
        agent.client.messages.create.assert_called_once()

    def test_tool_use_then_response(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        # Create a file so read_file works
        ws.root.mkdir(parents=True, exist_ok=True)
        (ws.root / "test.txt").write_text("file content")

        agent = StubAgent(workspace=ws)

        # First response: tool call to read a file
        tool_call = _make_tool_use_block("t1", "read_file", {"path": "test.txt"})
        first_response = _make_response([tool_call], stop_reason="tool_use")

        # Second response: final text
        second_response = _make_response([_make_text_block("Read the file.")])

        agent.client = MagicMock()
        agent.client.messages.create.side_effect = [first_response, second_response]

        task = Task(description="Read a file")
        result = agent.run(task)

        assert result.success
        assert "Read the file." in result.summary
        assert agent.client.messages.create.call_count == 2

    def test_unknown_tool(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        agent = StubAgent(workspace=ws)

        # Call an unknown tool, then finish
        tool_call = _make_tool_use_block("t1", "nonexistent_tool", {})
        first_response = _make_response([tool_call], stop_reason="tool_use")
        second_response = _make_response([_make_text_block("OK")])

        agent.client = MagicMock()
        agent.client.messages.create.side_effect = [first_response, second_response]

        task = Task(description="Try unknown tool")
        result = agent.run(task)

        assert result.success
        # Verify the error was passed back to the model.
        # messages is a shared mutable list; by the time we inspect it,
        # it has [user, assistant(tool_call), user(tool_result), assistant(text)].
        # The tool_result user message is at index 2.
        second_call_kwargs = agent.client.messages.create.call_args_list[1].kwargs
        all_messages = second_call_kwargs["messages"]
        tool_result_msg = all_messages[2]  # user message with tool_result
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["is_error"] is True

    def test_max_turns_exceeded(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        agent = StubAgent(workspace=ws)

        # Always return a tool call that never finishes
        tool_call = _make_tool_use_block("t1", "read_file", {"path": "x"})
        looping_response = _make_response([tool_call], stop_reason="tool_use")

        agent.client = MagicMock()
        agent.client.messages.create.return_value = looping_response

        # Write the file so it doesn't error
        (ws.root / "x").write_text("data")

        task = Task(description="Loop forever")
        result = agent.run(task)

        assert not result.success
        assert "exhausted" in result.summary.lower()

    def test_write_file_tool(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        agent = StubAgent(workspace=ws)

        tool_call = _make_tool_use_block(
            "t1", "write_file", {"path": "output.py", "content": "print('hi')"}
        )
        first_response = _make_response([tool_call], stop_reason="tool_use")
        second_response = _make_response([_make_text_block("Wrote the file.")])

        agent.client = MagicMock()
        agent.client.messages.create.side_effect = [first_response, second_response]

        task = Task(description="Write a file")
        result = agent.run(task)

        assert result.success
        assert (ws.root / "output.py").read_text() == "print('hi')"
