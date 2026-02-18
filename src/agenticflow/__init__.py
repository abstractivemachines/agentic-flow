"""AgenticFlow: Multi-agent software engineering system powered by Claude."""

from __future__ import annotations

from agenticflow.models import SharedContext, Task, TaskResult, Workspace

__all__ = [
    "ClaudeCodeOrchestrator",
    "LangGraphOrchestrator",
    "Orchestrator",
    "SharedContext",
    "StreamEvent",
    "Task",
    "TaskResult",
    "Workspace",
]


def __getattr__(name: str):
    if name == "Orchestrator" or name == "StreamEvent":
        from agenticflow.orchestrator import Orchestrator, StreamEvent

        return Orchestrator if name == "Orchestrator" else StreamEvent

    if name == "ClaudeCodeOrchestrator":
        from agenticflow.claude_code import ClaudeCodeOrchestrator

        return ClaudeCodeOrchestrator

    if name == "LangGraphOrchestrator":
        from agenticflow.langgraph_orchestrator import LangGraphOrchestrator

        return LangGraphOrchestrator

    raise AttributeError(f"module 'agenticflow' has no attribute {name!r}")
