"""Base tool interface for AgenticFlow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of executing a tool."""

    output: str
    is_error: bool = False

    def to_api_content(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": "text", "text": self.output}
        return result


class Tool(ABC):
    """Abstract base class for tools that agents can use."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (matches the function name Claude sees)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given arguments and return a result."""

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by the Anthropic API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
