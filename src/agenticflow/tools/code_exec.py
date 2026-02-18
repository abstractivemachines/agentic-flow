"""Code execution tool — runs shell commands in a subprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.tools.base import Tool, ToolResult
from agenticflow.tools.shell import ShellTool

DEFAULT_TIMEOUT = 30


class RunCommandTool(Tool):
    """Run a shell command in the workspace directory."""

    def __init__(self, workspace_root: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._shell = ShellTool(workspace_root, timeout=timeout)
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in the workspace directory and return stdout/stderr. "
            f"Commands time out after {self._timeout} seconds."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            "required": ["command"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]
        # Reuse ShellTool implementation so behavior (timeouts/output handling)
        # stays consistent across command-execution tools.
        return self._shell.execute(command=command)
