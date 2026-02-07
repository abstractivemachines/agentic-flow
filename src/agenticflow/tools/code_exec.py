"""Code execution tool — runs shell commands in a subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agenticflow.tools.base import Tool, ToolResult

DEFAULT_TIMEOUT = 30


class RunCommandTool(Tool):
    """Run a shell command in the workspace directory."""

    def __init__(self, workspace_root: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._workspace_root = workspace_root.resolve()
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
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._workspace_root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            parts = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr}")
            parts.append(f"Exit code: {result.returncode}")
            output = "\n".join(parts)
            return ToolResult(output=output, is_error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Command timed out after {self._timeout} seconds: {command}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)
