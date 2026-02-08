"""Tests for the LangGraph orchestrator backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

langgraph = pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agenticflow.langgraph_orchestrator import (  # noqa: E402
    APPROVAL_POLICIES,
    LangGraphOrchestrator,
    _get_workspace,
    _register_workspace,
    build_graph,
    default_approval_policy,
    evaluate_node,
    execute_agent_node,
    finalize_node,
    no_approval_policy,
    plan_node,
    revise_node,
    route_after_approval,
    route_after_evaluate,
    should_approve_node,
    strict_approval_policy,
)
from agenticflow.langgraph_state import (  # noqa: E402
    OrchestratorState,
    deserialize_task_result,
    serialize_task_result,
)
from agenticflow.models import AgentType, TaskResult, Workspace  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_response(
    content: list[Any], stop_reason: str = "end_turn"
) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _make_plan_response(plan: list[dict[str, str]]) -> SimpleNamespace:
    """Mock Claude response that returns a JSON plan."""
    return _make_response([_make_text_block(json.dumps(plan))])


def _make_agent_result(
    agent_type: str = "coder",
    success: bool = True,
    summary: str = "Done.",
    task_id: str = "abc123",
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        agent_type=AgentType(agent_type),
        success=success,
        summary=summary,
    )


def _make_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a minimal config dict for node functions."""
    ws = Workspace(root=tmp_path / "ws")
    key = _register_workspace(ws)
    configurable = {
        "workspace_key": key,
        "model": "test-model",
        "approval_policy": "none",
        "verbose": False,
        "thread_id": "test-thread",
    }
    configurable.update(overrides)
    return {"configurable": configurable}


def _make_state(**overrides: Any) -> OrchestratorState:
    """Build a base state with sensible defaults."""
    base: OrchestratorState = {
        "user_request": "Build something",
        "plan": [],
        "current_step_index": 0,
        "agent_results": [],
        "shared_context": {},
        "pending_approval": False,
        "approval_decision": None,
        "approval_feedback": None,
        "needs_revision": False,
        "revision_feedback": None,
        "final_summary": None,
        "is_complete": False,
        "error": None,
    }
    base.update(overrides)
    return base


# ===========================================================================
# State serialization tests
# ===========================================================================


class TestStateSerialization:
    def test_serialize_task_result(self):
        result = _make_agent_result(summary="Wrote hello.py")
        data = serialize_task_result(result)
        assert data["agent_type"] == "coder"
        assert data["success"] is True
        assert data["summary"] == "Wrote hello.py"
        assert isinstance(data["artifacts"], dict)

    def test_deserialize_task_result(self):
        data = {
            "task_id": "t1",
            "agent_type": "reviewer",
            "success": False,
            "summary": "REQUEST_CHANGES",
            "artifacts": {},
            "errors": ["issue1"],
        }
        result = deserialize_task_result(data)
        assert result.agent_type == AgentType.REVIEWER
        assert result.success is False
        assert result.errors == ["issue1"]

    def test_roundtrip(self):
        original = _make_agent_result(agent_type="tester", summary="All pass")
        data = serialize_task_result(original)
        restored = deserialize_task_result(data)
        assert restored.task_id == original.task_id
        assert restored.agent_type == original.agent_type
        assert restored.summary == original.summary


# ===========================================================================
# Approval policy tests
# ===========================================================================


class TestApprovalPolicies:
    def test_default_gates_coder(self):
        assert default_approval_policy("coder") is True
        assert default_approval_policy("planner") is False
        assert default_approval_policy("reviewer") is False

    def test_strict_gates_all(self):
        assert strict_approval_policy("coder") is True
        assert strict_approval_policy("planner") is True

    def test_none_gates_nothing(self):
        assert no_approval_policy("coder") is False
        assert no_approval_policy("planner") is False

    def test_policies_dict(self):
        assert "default" in APPROVAL_POLICIES
        assert "strict" in APPROVAL_POLICIES
        assert "none" in APPROVAL_POLICIES


# ===========================================================================
# Workspace registry tests
# ===========================================================================


class TestWorkspaceRegistry:
    def test_register_and_get(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws-reg")
        key = _register_workspace(ws)
        config = {"configurable": {"workspace_key": key}}
        assert _get_workspace(config) is ws

    def test_get_missing_raises(self):
        config = {"configurable": {"workspace_key": "nonexistent"}}
        with pytest.raises(RuntimeError, match="Workspace not found"):
            _get_workspace(config)


# ===========================================================================
# Node unit tests
# ===========================================================================


class TestPlanNode:
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_successful_plan(self, mock_anthropic_cls, tmp_path: Path):
        plan = [
            {"agent_type": "coder", "task_description": "Write hello.py"},
            {"agent_type": "tester", "task_description": "Test hello.py"},
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        state = _make_state(user_request="Create hello world")
        config = _make_config(tmp_path)
        result = plan_node(state, config)

        assert result["plan"] == plan
        assert result["current_step_index"] == 0
        assert result["agent_results"] == []

    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_invalid_json_plan(self, mock_anthropic_cls, tmp_path: Path):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            [_make_text_block("This is not JSON")]
        )
        mock_anthropic_cls.return_value = mock_client

        state = _make_state()
        config = _make_config(tmp_path)
        result = plan_node(state, config)

        assert "error" in result
        assert result["is_complete"] is True

    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_non_array_json(self, mock_anthropic_cls, tmp_path: Path):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            [_make_text_block('{"not": "an array"}')]
        )
        mock_anthropic_cls.return_value = mock_client

        state = _make_state()
        config = _make_config(tmp_path)
        result = plan_node(state, config)

        assert "error" in result
        assert "must be a JSON array" in result["error"]


class TestShouldApproveNode:
    def test_no_approval_needed(self, tmp_path: Path):
        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "code"}],
        )
        config = _make_config(tmp_path, approval_policy="none")
        result = should_approve_node(state, config)

        assert result["pending_approval"] is False
        assert result["approval_decision"] == "approve"

    def test_index_out_of_bounds(self, tmp_path: Path):
        state = _make_state(plan=[], current_step_index=0)
        config = _make_config(tmp_path)
        result = should_approve_node(state, config)
        assert result["pending_approval"] is False


class TestExecuteAgentNode:
    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    def test_executes_agent(self, mock_classes, tmp_path: Path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Code written")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Write code"}],
        )
        config = _make_config(tmp_path)
        result = execute_agent_node(state, config)

        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["summary"] == "Code written"
        assert result["needs_revision"] is False
        mock_agent.run.assert_called_once()

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    def test_includes_shared_context(self, mock_classes, tmp_path: Path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result()
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Do stuff"}],
            shared_context={"plan": "Build REST API"},
        )
        config = _make_config(tmp_path)
        execute_agent_node(state, config)

        # Verify the task description includes context
        call_args = mock_agent.run.call_args[0][0]
        assert "Previous context" in call_args.description
        assert "Build REST API" in call_args.description

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    def test_appends_to_existing_results(self, mock_classes, tmp_path: Path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Step 2 done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        existing_result = serialize_task_result(
            _make_agent_result(summary="Step 1 done")
        )
        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Step 2"}],
            agent_results=[existing_result],
        )
        config = _make_config(tmp_path)
        result = execute_agent_node(state, config)

        assert len(result["agent_results"]) == 2


class TestEvaluateNode:
    def test_advance_to_next_step(self):
        state = _make_state(
            plan=[
                {"agent_type": "coder", "task_description": "Code"},
                {"agent_type": "tester", "task_description": "Test"},
            ],
            current_step_index=0,
            agent_results=[{"success": True, "summary": "Done"}],
        )
        result = evaluate_node(state)
        assert result["current_step_index"] == 1
        assert "is_complete" not in result

    def test_complete_when_last_step(self):
        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Code"}],
            current_step_index=0,
            agent_results=[{"success": True, "summary": "Done"}],
        )
        result = evaluate_node(state)
        assert result["is_complete"] is True
        assert result["current_step_index"] == 1

    def test_revision_on_coder_failure(self):
        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Code"}],
            current_step_index=0,
            agent_results=[{"success": False, "summary": "Syntax error"}],
        )
        result = evaluate_node(state)
        assert result["needs_revision"] is True
        assert "Syntax error" in result["revision_feedback"]

    def test_revision_on_reviewer_request_changes(self):
        state = _make_state(
            plan=[
                {"agent_type": "coder", "task_description": "Code"},
                {"agent_type": "reviewer", "task_description": "Review"},
            ],
            current_step_index=1,
            agent_results=[
                {"success": True, "summary": "Done"},
                {"success": True, "summary": "REQUEST_CHANGES: Fix imports"},
            ],
        )
        result = evaluate_node(state)
        assert result["needs_revision"] is True
        assert result["current_step_index"] == 0  # back to coder

    def test_no_results_error(self):
        state = _make_state(agent_results=[])
        result = evaluate_node(state)
        assert result["is_complete"] is True
        assert "No results" in result["error"]


class TestReviseNode:
    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    def test_dispatches_coder_with_feedback(self, mock_classes, tmp_path: Path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Fixed")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        state = _make_state(
            plan=[{"agent_type": "coder", "task_description": "Write code"}],
            revision_feedback="Fix the bug in line 10",
        )
        config = _make_config(tmp_path)
        result = revise_node(state, config)

        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["summary"] == "Fixed"
        assert result["needs_revision"] is False

        # Verify feedback was included in the task
        task_arg = mock_agent.run.call_args[0][0]
        assert "Fix the bug in line 10" in task_arg.description


class TestFinalizeNode:
    def test_with_results(self):
        state = _make_state(
            agent_results=[
                {"success": True, "agent_type": "coder", "summary": "Wrote code"},
                {"success": True, "agent_type": "tester", "summary": "Tests pass"},
            ],
        )
        result = finalize_node(state)
        assert "Orchestration complete" in result["final_summary"]
        assert "Wrote code" in result["final_summary"]
        assert result["is_complete"] is True

    def test_with_error(self):
        state = _make_state(error="Plan parsing failed")
        result = finalize_node(state)
        assert "Error: Plan parsing failed" in result["final_summary"]

    def test_no_results(self):
        state = _make_state()
        result = finalize_node(state)
        assert "No work was performed" in result["final_summary"]

    def test_rejection_noted(self):
        state = _make_state(
            agent_results=[
                {"success": True, "agent_type": "coder", "summary": "Done"},
            ],
            approval_decision="reject",
            approval_feedback="Not what I wanted",
        )
        result = finalize_node(state)
        assert "rejected" in result["final_summary"]
        assert "Not what I wanted" in result["final_summary"]
        assert "Partial results" in result["final_summary"]

    def test_rejection_no_results(self):
        state = _make_state(
            approval_decision="reject",
            approval_feedback="Too risky",
        )
        result = finalize_node(state)
        assert "rejected" in result["final_summary"]
        assert "Too risky" in result["final_summary"]


# ===========================================================================
# Routing function tests
# ===========================================================================


class TestRouting:
    def test_route_after_approval_approve(self):
        state = _make_state(approval_decision="approve")
        assert route_after_approval(state) == "execute_agent"

    def test_route_after_approval_reject(self):
        state = _make_state(approval_decision="reject")
        assert route_after_approval(state) == "finalize"

    def test_route_after_approval_default(self):
        state = _make_state()
        assert route_after_approval(state) == "execute_agent"

    def test_route_after_evaluate_revise(self):
        state = _make_state(needs_revision=True)
        assert route_after_evaluate(state) == "revise"

    def test_route_after_evaluate_finalize(self):
        state = _make_state(is_complete=True)
        assert route_after_evaluate(state) == "finalize"

    def test_route_after_evaluate_continue(self):
        state = _make_state(needs_revision=False, is_complete=False)
        assert route_after_evaluate(state) == "should_approve"


# ===========================================================================
# Graph builder test
# ===========================================================================


class TestBuildGraph:
    def test_builds_graph(self):
        builder = build_graph()
        assert set(builder.nodes.keys()) == {
            "plan",
            "should_approve",
            "execute_agent",
            "evaluate",
            "revise",
            "finalize",
        }

    def test_compiles_with_checkpointer(self):
        builder = build_graph()
        compiled = builder.compile(checkpointer=MemorySaver())
        assert compiled is not None


# ===========================================================================
# Full graph flow tests (mocked nodes, real graph)
# ===========================================================================


class TestGraphFlow:
    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_simple_plan_execute_finalize(
        self, mock_anthropic_cls, mock_classes, tmp_path: Path
    ):
        """Test a simple 1-step plan→execute→finalize flow."""
        plan = [{"agent_type": "coder", "task_description": "Write hello.py"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="hello.py created")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        checkpointer = MemorySaver()
        compiled = build_graph().compile(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": "test-simple",
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "none",
                "verbose": False,
            }
        }

        result = compiled.invoke(_make_state(user_request="Create hello world"), config)
        assert result["is_complete"] is True
        assert result["final_summary"] is not None
        assert "hello.py created" in result["final_summary"]

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_multi_step_flow(self, mock_anthropic_cls, mock_classes, tmp_path: Path):
        """Test a 2-step plan with coder + tester."""
        plan = [
            {"agent_type": "coder", "task_description": "Write code"},
            {"agent_type": "tester", "task_description": "Write tests"},
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            _make_agent_result(agent_type="coder", summary="Code written"),
            _make_agent_result(agent_type="tester", summary="Tests pass"),
        ]
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        compiled = build_graph().compile(checkpointer=MemorySaver())

        config = {
            "configurable": {
                "thread_id": "test-multi",
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "none",
                "verbose": False,
            }
        }

        result = compiled.invoke(_make_state(user_request="Build something"), config)
        assert result["is_complete"] is True
        assert len(result["agent_results"]) == 2

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_revision_cycle(self, mock_anthropic_cls, mock_classes, tmp_path: Path):
        """Test that a failed coder triggers revision."""
        plan = [{"agent_type": "coder", "task_description": "Write code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            _make_agent_result(success=False, summary="Syntax error in main.py"),
            _make_agent_result(success=True, summary="Fixed and working"),
        ]
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        compiled = build_graph().compile(checkpointer=MemorySaver())

        config = {
            "configurable": {
                "thread_id": "test-revision",
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "none",
                "verbose": False,
            }
        }

        result = compiled.invoke(_make_state(user_request="Build something"), config)
        assert result["is_complete"] is True
        # Should have 2 results: original failed + revision succeeded
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][1]["success"] is True

    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_plan_parse_error_flow(self, mock_anthropic_cls, tmp_path: Path):
        """Test that a plan parse error goes straight to finalize."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            [_make_text_block("not json at all")]
        )
        mock_anthropic_cls.return_value = mock_client

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        compiled = build_graph().compile(checkpointer=MemorySaver())

        config = {
            "configurable": {
                "thread_id": "test-parse-error",
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "none",
                "verbose": False,
            }
        }

        result = compiled.invoke(_make_state(user_request="Do something"), config)
        assert result["is_complete"] is True
        assert result["error"] is not None
        assert "Failed to parse" in result["error"]


# ===========================================================================
# Checkpoint tests
# ===========================================================================


class TestCheckpoint:
    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_state_persisted_across_invocations(
        self, mock_anthropic_cls, mock_classes, tmp_path: Path
    ):
        """Verify that state is checkpointed and can be retrieved."""
        plan = [{"agent_type": "coder", "task_description": "Code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        checkpointer = MemorySaver()
        compiled = build_graph().compile(checkpointer=checkpointer)

        thread_id = "test-checkpoint"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "none",
                "verbose": False,
            }
        }

        compiled.invoke(_make_state(user_request="Test checkpoint"), config)

        # Retrieve state
        state_snapshot = compiled.get_state(config)
        assert state_snapshot.values["is_complete"] is True

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_interrupt_and_resume(
        self, mock_anthropic_cls, mock_classes, tmp_path: Path
    ):
        """Test that interrupt() pauses and Command(resume=...) continues."""
        plan = [{"agent_type": "coder", "task_description": "Code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Code done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        checkpointer = MemorySaver()
        compiled = build_graph().compile(checkpointer=checkpointer)

        thread_id = "test-interrupt"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "strict",  # gates all agents
                "verbose": False,
            }
        }

        # First invoke — should interrupt at should_approve
        result = compiled.invoke(_make_state(user_request="Build it"), config)

        # Check that it was interrupted
        graph_state = compiled.get_state(config)
        assert len(graph_state.tasks) > 0  # Has pending interrupt

        # Resume with approval
        decision = {"decision": "approve", "feedback": None}
        result = compiled.invoke(Command(resume=decision), config)

        assert result["is_complete"] is True
        assert "Code done" in result["final_summary"]

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_interrupt_and_reject(
        self, mock_anthropic_cls, mock_classes, tmp_path: Path
    ):
        """Test that rejecting at interrupt skips to finalize."""
        plan = [{"agent_type": "coder", "task_description": "Code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        ws = Workspace(root=tmp_path / "ws")
        key = _register_workspace(ws)
        checkpointer = MemorySaver()
        compiled = build_graph().compile(checkpointer=checkpointer)

        thread_id = "test-reject"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "workspace_key": key,
                "model": "test-model",
                "approval_policy": "strict",
                "verbose": False,
            }
        }

        # First invoke — interrupts
        compiled.invoke(_make_state(user_request="Build it"), config)

        # Resume with rejection
        decision = {"decision": "reject", "feedback": "Too risky"}
        result = compiled.invoke(Command(resume=decision), config)

        assert result["is_complete"] is True
        assert "rejected" in result["final_summary"]
        assert "Too risky" in result["final_summary"]


# ===========================================================================
# LangGraphOrchestrator class tests
# ===========================================================================


class TestLangGraphOrchestrator:
    def test_init_memory_backend(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = LangGraphOrchestrator(
            workspace=ws,
            checkpoint_backend="memory",
            approval_policy="none",
        )
        assert orch.approval_policy == "none"
        assert orch._graph is not None

    def test_init_sqlite_backend(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        LangGraphOrchestrator(
            workspace=ws,
            checkpoint_backend="sqlite",
        )
        db_path = ws.root / ".agenticflow" / "checkpoints.db"
        assert db_path.parent.exists()

    def test_init_invalid_backend(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        with pytest.raises(ValueError, match="Unknown checkpoint backend"):
            LangGraphOrchestrator(workspace=ws, checkpoint_backend="redis")

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_run(self, mock_anthropic_cls, mock_classes, tmp_path: Path):
        plan = [{"agent_type": "coder", "task_description": "Write code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Code done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        orch = LangGraphOrchestrator(
            workspace=ws,
            checkpoint_backend="memory",
            approval_policy="none",
        )
        result = orch.run("Write hello world")
        assert "Code done" in result

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_run_stream(self, mock_anthropic_cls, mock_classes, tmp_path: Path):
        plan = [{"agent_type": "coder", "task_description": "Write code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Code done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        orch = LangGraphOrchestrator(
            workspace=ws,
            checkpoint_backend="memory",
            approval_policy="none",
        )
        events = list(orch.run_stream("Write hello world"))
        kinds = {e.kind for e in events}
        assert "done" in kinds

    @patch("agenticflow.langgraph_orchestrator.AGENT_CLASSES")
    @patch("agenticflow.langgraph_orchestrator.anthropic.Anthropic")
    def test_get_state(self, mock_anthropic_cls, mock_classes, tmp_path: Path):
        plan = [{"agent_type": "coder", "task_description": "Write code"}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_plan_response(plan)
        mock_anthropic_cls.return_value = mock_client

        mock_agent = MagicMock()
        mock_agent.run.return_value = _make_agent_result(summary="Done")
        mock_cls = MagicMock(return_value=mock_agent)
        mock_classes.__getitem__ = MagicMock(return_value=mock_cls)

        ws = Workspace(root=tmp_path / "ws")
        orch = LangGraphOrchestrator(
            workspace=ws,
            checkpoint_backend="memory",
            approval_policy="none",
        )
        orch.run("Test get_state", thread_id="state-thread")
        state = orch.get_state("state-thread")
        assert state.values["is_complete"] is True
