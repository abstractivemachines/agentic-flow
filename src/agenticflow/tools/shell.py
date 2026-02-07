"""Shell tool — general command execution with configurable timeout and env."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agenticflow.tools.base import Tool, ToolResult

DEFAULT_TIMEOUT = 60
MAX_OUTPUT_CHARS = 50_000


class ShellTool(Tool):
    """Execute arbitrary shell commands with configurable timeout."""

    def __init__(self, workspace_root: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._workspace_root = workspace_root.resolve()
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command in the workspace directory. "
            "Supports an optional timeout override and environment variables. "
            f"Default timeout: {self._timeout}s."
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
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {self._timeout}).",
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables to set.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["command"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]
        timeout = kwargs.get("timeout", self._timeout)
        extra_env = kwargs.get("env")

        import os
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            parts = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr}")
            parts.append(f"Exit code: {result.returncode}")
            output = "\n".join(parts)

            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"

            return ToolResult(output=output, is_error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Command timed out after {timeout} seconds: {command}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)
