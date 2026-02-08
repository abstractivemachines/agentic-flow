"""Tests for the Claude Code CLI backend — all SDK calls mocked."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenticflow.models import AgentType, Workspace


# ---------------------------------------------------------------------------
# Mock SDK types — proper classes so isinstance() works
# ---------------------------------------------------------------------------


class _MockTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockAssistantMessage:
    def __init__(self, content: list[Any], model: str = "test-model") -> None:
        self.content = content
        self.model = model


class _MockResultMessage:
    def __init__(
        self,
        result: str | None = None,
        is_error: bool = False,
    ) -> None:
        self.subtype = "result"
        self.duration_ms = 100
        self.duration_api_ms = 80
        self.is_error = is_error
        self.num_turns = 1
        self.session_id = "test-session"
        self.total_cost_usd = 0.01
        self.usage = {}
        self.result = result


def _make_assistant_message(text: str) -> _MockAssistantMessage:
    return _MockAssistantMessage(content=[_MockTextBlock(text)])


def _make_result_message(
    result: str | None = None,
    is_error: bool = False,
) -> _MockResultMessage:
    return _MockResultMessage(result=result, is_error=is_error)


def _build_mock_sdk(query_messages: list[Any] | None = None) -> MagicMock:
    """Build a mock claude_agent_sdk module with commonly needed attributes."""
    sdk = MagicMock()

    # Types that isinstance checks need — use our real mock classes
    sdk.AssistantMessage = _MockAssistantMessage
    sdk.TextBlock = _MockTextBlock
    sdk.ResultMessage = _MockResultMessage

    # ClaudeAgentOptions as a simple namespace factory
    sdk.ClaudeAgentOptions = lambda **kwargs: SimpleNamespace(**kwargs)

    # Tool decorator: return the function wrapped
    def mock_tool(name, description, schema):
        def decorator(fn):
            fn._tool_name = name
            return fn

        return decorator

    sdk.tool = mock_tool

    # create_sdk_mcp_server: return a sentinel
    sdk.create_sdk_mcp_server = MagicMock(
        return_value=SimpleNamespace(name="agenticflow")
    )

    # query: async generator yielding provided messages
    if query_messages is not None:

        async def mock_query(**kwargs):
            for msg in query_messages:
                yield msg

        sdk.query = mock_query

    return sdk


# ---------------------------------------------------------------------------
# Tests for _import_sdk
# ---------------------------------------------------------------------------


class TestImportSdk:
    def test_import_error_without_sdk(self):
        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            from agenticflow.claude_code import _import_sdk

            with pytest.raises(ImportError, match="claude-agent-sdk is required"):
                _import_sdk()

    def test_import_succeeds_with_sdk(self):
        mock_sdk = MagicMock()
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            from agenticflow.claude_code import _import_sdk

            result = _import_sdk()
            assert result is mock_sdk


# ---------------------------------------------------------------------------
# Tests for CLAUDE_CODE_TOOL_MAPPING
# ---------------------------------------------------------------------------


class TestToolMapping:
    def test_tool_mapping_completeness(self):
        from agenticflow.claude_code import CLAUDE_CODE_TOOL_MAPPING

        for agent_type in AgentType:
            assert agent_type in CLAUDE_CODE_TOOL_MAPPING
            assert len(CLAUDE_CODE_TOOL_MAPPING[agent_type]) > 0

    def test_coder_has_write_tools(self):
        from agenticflow.claude_code import CLAUDE_CODE_TOOL_MAPPING

        coder_tools = CLAUDE_CODE_TOOL_MAPPING[AgentType.CODER]
        assert "Write" in coder_tools
        assert "Edit" in coder_tools

    def test_reviewer_has_no_write_tools(self):
        from agenticflow.claude_code import CLAUDE_CODE_TOOL_MAPPING

        reviewer_tools = CLAUDE_CODE_TOOL_MAPPING[AgentType.REVIEWER]
        assert "Write" not in reviewer_tools
        assert "Edit" not in reviewer_tools


# ---------------------------------------------------------------------------
# Tests for _build_agent_system_prompt
# ---------------------------------------------------------------------------


class TestBuildAgentSystemPrompt:
    def test_returns_preset_dict(self, tmp_path: Path):
        from agenticflow.claude_code import _build_agent_system_prompt

        ws = Workspace(root=tmp_path / "ws")
        result = _build_agent_system_prompt(AgentType.PLANNER, ws)

        assert result["type"] == "preset"
        assert result["preset"] == "claude_code"
        assert isinstance(result["append"], str)
        assert "architect" in result["append"].lower()

    def test_includes_shared_context(self, tmp_path: Path):
        from agenticflow.claude_code import _build_agent_system_prompt

        ws = Workspace(root=tmp_path / "ws")
        ws.shared_context.set("plan", "Build a REST API")

        result = _build_agent_system_prompt(AgentType.CODER, ws)
        assert "Build a REST API" in result["append"]

    def test_includes_tool_note(self, tmp_path: Path):
        from agenticflow.claude_code import _build_agent_system_prompt

        ws = Workspace(root=tmp_path / "ws")
        result = _build_agent_system_prompt(AgentType.CODER, ws)
        assert "Claude Code native tools" in result["append"]


# ---------------------------------------------------------------------------
# Tests for run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    def test_run_agent_success(self, tmp_path: Path):
        from agenticflow.claude_code import run_agent

        ws = Workspace(root=tmp_path / "ws")

        assistant_msg = _make_assistant_message("Code written successfully.")
        result_msg = _make_result_message(result="All done.")
        mock_sdk = _build_mock_sdk([assistant_msg, result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            task_result = asyncio.run(run_agent(AgentType.CODER, "Write hello.py", ws))

        assert task_result.success is True
        assert task_result.agent_type == AgentType.CODER
        assert "All done." in task_result.summary

    def test_run_agent_failure(self, tmp_path: Path):
        from agenticflow.claude_code import run_agent

        ws = Workspace(root=tmp_path / "ws")

        mock_sdk = _build_mock_sdk()

        async def failing_query(**kwargs):
            raise RuntimeError("CLI crashed")
            yield  # noqa: F841 — unreachable but makes this an async generator

        mock_sdk.query = failing_query

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            task_result = asyncio.run(run_agent(AgentType.CODER, "Write hello.py", ws))

        assert task_result.success is False
        assert "CLI crashed" in task_result.summary

    def test_run_agent_error_result(self, tmp_path: Path):
        from agenticflow.claude_code import run_agent

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Permission denied", is_error=True)
        mock_sdk = _build_mock_sdk([result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            task_result = asyncio.run(
                run_agent(AgentType.PLANNER, "Plan something", ws)
            )

        assert task_result.success is False
        assert "Permission denied" in task_result.summary

    def test_run_agent_passes_add_dirs(self, tmp_path: Path):
        from agenticflow.claude_code import run_agent

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Done.")
        mock_sdk = _build_mock_sdk([result_msg])

        captured_options = []
        original_factory = mock_sdk.ClaudeAgentOptions

        def capturing_factory(**kwargs):
            opts = original_factory(**kwargs)
            captured_options.append(kwargs)
            return opts

        mock_sdk.ClaudeAgentOptions = capturing_factory

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            asyncio.run(
                run_agent(
                    AgentType.CODER,
                    "Write code",
                    ws,
                    add_dirs=["/home/user/project"],
                )
            )

        assert len(captured_options) == 1
        assert captured_options[0]["add_dirs"] == ["/home/user/project"]

    def test_run_agent_omits_add_dirs_when_none(self, tmp_path: Path):
        from agenticflow.claude_code import run_agent

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Done.")
        mock_sdk = _build_mock_sdk([result_msg])

        captured_options = []
        original_factory = mock_sdk.ClaudeAgentOptions

        def capturing_factory(**kwargs):
            opts = original_factory(**kwargs)
            captured_options.append(kwargs)
            return opts

        mock_sdk.ClaudeAgentOptions = capturing_factory

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            asyncio.run(run_agent(AgentType.CODER, "Write code", ws))

        assert len(captured_options) == 1
        assert "add_dirs" not in captured_options[0]


# ---------------------------------------------------------------------------
# Tests for ClaudeCodeOrchestrator
# ---------------------------------------------------------------------------


class TestClaudeCodeOrchestrator:
    def test_orchestrator_run(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Project complete.")
        mock_sdk = _build_mock_sdk([result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)
            result = asyncio.run(orch.run("Build a web app"))

        assert result == "Project complete."

    def test_orchestrator_run_with_assistant_message(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        assistant_msg = _make_assistant_message("Here is the summary.")
        result_msg = _make_result_message(result=None)
        mock_sdk = _build_mock_sdk([assistant_msg, result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)
            result = asyncio.run(orch.run("Build something"))

        assert result == "Here is the summary."

    def test_orchestrator_run_stream(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        assistant_msg = _make_assistant_message("Streaming text.")
        result_msg = _make_result_message(result="Final.")
        mock_sdk = _build_mock_sdk([assistant_msg, result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)

            async def collect_events():
                events = []
                async for event in orch.run_stream("Do something"):
                    events.append(event)
                return events

            events = asyncio.run(collect_events())

        assert any(e.kind == "text" for e in events)
        assert any(e.kind == "done" for e in events)

    def test_orchestrator_run_stream_error(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Something broke", is_error=True)
        mock_sdk = _build_mock_sdk([result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)

            async def collect_events():
                events = []
                async for event in orch.run_stream("Fail"):
                    events.append(event)
                return events

            events = asyncio.run(collect_events())

        assert any(e.kind == "error" for e in events)

    def test_orchestrator_run_sync(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Sync result.")
        mock_sdk = _build_mock_sdk([result_msg])

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)
            result = orch.run_sync("Quick task")

        assert result == "Sync result."

    def test_orchestrator_passes_add_dirs_with_invocation_dir(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")
        inv_dir = tmp_path / "project"
        inv_dir.mkdir()

        result_msg = _make_result_message(result="Done.")
        mock_sdk = _build_mock_sdk([result_msg])

        captured_options = []
        original_factory = mock_sdk.ClaudeAgentOptions

        def capturing_factory(**kwargs):
            opts = original_factory(**kwargs)
            captured_options.append(kwargs)
            return opts

        mock_sdk.ClaudeAgentOptions = capturing_factory

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws, invocation_dir=inv_dir)
            asyncio.run(orch.run("Build something"))

        assert len(captured_options) == 1
        assert captured_options[0]["add_dirs"] == [str(inv_dir.resolve())]

    def test_orchestrator_no_add_dirs_when_same_as_workspace(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws_dir = tmp_path / "ws"
        ws_dir.mkdir()
        ws = Workspace(root=ws_dir)

        result_msg = _make_result_message(result="Done.")
        mock_sdk = _build_mock_sdk([result_msg])

        captured_options = []
        original_factory = mock_sdk.ClaudeAgentOptions

        def capturing_factory(**kwargs):
            opts = original_factory(**kwargs)
            captured_options.append(kwargs)
            return opts

        mock_sdk.ClaudeAgentOptions = capturing_factory

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws, invocation_dir=ws_dir)
            asyncio.run(orch.run("Build something"))

        assert len(captured_options) == 1
        assert "add_dirs" not in captured_options[0]

    def test_orchestrator_no_add_dirs_without_invocation_dir(self, tmp_path: Path):
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        ws = Workspace(root=tmp_path / "ws")

        result_msg = _make_result_message(result="Done.")
        mock_sdk = _build_mock_sdk([result_msg])

        captured_options = []
        original_factory = mock_sdk.ClaudeAgentOptions

        def capturing_factory(**kwargs):
            opts = original_factory(**kwargs)
            captured_options.append(kwargs)
            return opts

        mock_sdk.ClaudeAgentOptions = capturing_factory

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            orch = ClaudeCodeOrchestrator(workspace=ws)
            asyncio.run(orch.run("Build something"))

        assert len(captured_options) == 1
        assert "add_dirs" not in captured_options[0]


# ---------------------------------------------------------------------------
# Tests for MCP tool handlers
# ---------------------------------------------------------------------------


class TestMcpTools:
    def test_mcp_dispatch_tool(self, tmp_path: Path):
        from agenticflow.claude_code import _build_orchestrator_mcp_server
        from agenticflow.models import TaskResult

        ws = Workspace(root=tmp_path / "ws")

        mock_sdk = _build_mock_sdk()

        # Capture the decorated functions
        tool_fns = {}

        def capturing_tool(name, description, schema):
            def decorator(fn):
                tool_fns[name] = fn
                fn._tool_name = name
                return fn

            return decorator

        mock_sdk.tool = capturing_tool

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            _build_orchestrator_mcp_server(ws, {}, None, False)

        assert "dispatch_agent" in tool_fns
        assert "set_shared_context" in tool_fns

        # Test dispatch_agent handler
        mock_result = TaskResult(
            task_id="test123",
            agent_type=AgentType.CODER,
            success=True,
            summary="Code written.",
        )

        with patch(
            "agenticflow.claude_code.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result

            with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
                result = asyncio.run(
                    tool_fns["dispatch_agent"](
                        {
                            "agent_type": "coder",
                            "task_description": "Write hello.py",
                        }
                    )
                )

        assert "SUCCESS" in result["content"][0]["text"]
        assert "Code written." in result["content"][0]["text"]
        assert len(ws.task_results) == 1
        assert ws.task_results[0].summary == "Code written."

    def test_mcp_set_context_tool(self, tmp_path: Path):
        from agenticflow.claude_code import _build_orchestrator_mcp_server

        ws = Workspace(root=tmp_path / "ws")

        mock_sdk = _build_mock_sdk()
        tool_fns = {}

        def capturing_tool(name, description, schema):
            def decorator(fn):
                tool_fns[name] = fn
                fn._tool_name = name
                return fn

            return decorator

        mock_sdk.tool = capturing_tool

        with patch("agenticflow.claude_code._import_sdk", return_value=mock_sdk):
            _build_orchestrator_mcp_server(ws, {}, None, False)

        result = asyncio.run(
            tool_fns["set_shared_context"](
                {
                    "key": "plan",
                    "value": "Build a REST API",
                }
            )
        )

        assert "Stored context key 'plan'" in result["content"][0]["text"]
        assert ws.shared_context.get("plan") == "Build a REST API"
