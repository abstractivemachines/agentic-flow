"""Core data models for AgenticFlow."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(Enum):
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"


@dataclass
class Task:
    """A unit of work to be performed by an agent."""

    description: str
    agent_type: AgentType | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    parent_task_id: str | None = None

    def to_prompt(self) -> str:
        parts = [f"Task: {self.description}"]
        if self.context:
            parts.append(f"Context: {self.context}")
        return "\n".join(parts)


@dataclass
class TaskResult:
    """Result from an agent completing a task."""

    task_id: str
    agent_type: AgentType
    success: bool
    summary: str
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class Workspace:
    """Shared workspace for agents — backed by a physical directory and in-memory metadata."""

    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path within the workspace, preventing traversal."""
        resolved = (self.root / relative_path).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError(f"Path traversal detected: {relative_path}")
        return resolved

    def add_result(self, result: TaskResult) -> None:
        self.task_results.append(result)

    def get_results_summary(self) -> str:
        if not self.task_results:
            return "No completed tasks yet."
        lines = []
        for r in self.task_results:
            status = "SUCCESS" if r.success else "FAILED"
            lines.append(f"[{status}] {r.agent_type.value}: {r.summary}")
        return "\n".join(lines)
