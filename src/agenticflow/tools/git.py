"""Git tool — agents can run git operations in the workspace."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from agenticflow.tools.base import Tool, ToolResult

ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "add", "commit", "branch", "checkout",
    "merge", "stash", "show", "init", "reset", "tag",
}

DEFAULT_TIMEOUT = 30


class GitTool(Tool):
    """Run git commands in the workspace directory."""

    def __init__(self, workspace_root: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._workspace_root = workspace_root.resolve()
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run a git command in the workspace. Provide the subcommand and arguments "
            "as a single string (e.g. 'status', 'add .', 'commit -m \"message\"'). "
            f"Allowed subcommands: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "string",
                    "description": (
                        "Git arguments (e.g. 'status', 'add .', "
                        "'commit -m \"initial commit\"')."
                    ),
                },
            },
            "required": ["args"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        args = kwargs["args"].strip()
        if not args:
            return ToolResult(output="No git arguments provided.", is_error=True)

        try:
            argv = shlex.split(args)
        except ValueError as exc:
            return ToolResult(output=f"Invalid git arguments: {exc}", is_error=True)

        if not argv:
            return ToolResult(output="No git arguments provided.", is_error=True)

        subcommand = argv[0]
        if subcommand not in ALLOWED_SUBCOMMANDS:
            return ToolResult(
                output=f"Git subcommand '{subcommand}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}",
                is_error=True,
            )

        try:
            result = subprocess.run(
                ["git", *argv],
                cwd=self._workspace_root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            parts = []
            if result.stdout:
                parts.append(result.stdout)
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr}")
            parts.append(f"Exit code: {result.returncode}")
            output = "\n".join(parts)
            return ToolResult(output=output, is_error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Git command timed out after {self._timeout} seconds.",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)
