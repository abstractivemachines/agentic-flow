"""LangGraph state definition for the orchestrator graph.

Kept separate for clean imports and LangGraph Studio compatibility.
All values are JSON-serializable primitives so checkpointing works.
"""

from __future__ import annotations

from typing import Any, TypedDict

from agenticflow.models import AgentType, TaskResult


class OrchestratorState(TypedDict, total=False):
    """Full state for the LangGraph orchestrator graph."""

    # Core fields
    user_request: str
    plan: list[dict[str, str]]  # [{agent_type, task_description}, ...]
    current_step_index: int
    agent_results: list[dict[str, Any]]
    shared_context: dict[str, str]

    # Human-in-the-loop fields
    pending_approval: bool
    approval_decision: str | None  # "approve" | "reject"
    approval_feedback: str | None

    # Revision loop fields
    needs_revision: bool
    revision_feedback: str | None

    # Completion fields
    final_summary: str | None
    is_complete: bool
    error: str | None


def serialize_task_result(result: TaskResult) -> dict[str, Any]:
    """Convert a TaskResult to a JSON-serializable dict."""
    return {
        "task_id": result.task_id,
        "agent_type": result.agent_type.value,
        "success": result.success,
        "summary": result.summary,
        "artifacts": result.artifacts,
        "errors": result.errors,
    }


def deserialize_task_result(data: dict[str, Any]) -> TaskResult:
    """Reconstruct a TaskResult from a serialized dict."""
    return TaskResult(
        task_id=data["task_id"],
        agent_type=AgentType(data["agent_type"]),
        success=data["success"],
        summary=data["summary"],
        artifacts=data.get("artifacts", {}),
        errors=data.get("errors", []),
    )
