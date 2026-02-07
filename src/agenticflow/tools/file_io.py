"""File I/O tools for agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.tools.base import Tool, ToolResult


class ReadFileTool(Tool):
    """Read the contents of a file in the workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Provide a path relative to the workspace root."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace.",
                },
            },
            "required": ["path"],
        }

    def _resolve(self, path: str) -> Path:
        resolved = (self._workspace_root / path).resolve()
        if not str(resolved).startswith(str(self._workspace_root)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs["path"]
        try:
            resolved = self._resolve(path)
            if not resolved.exists():
                return ToolResult(output=f"File not found: {path}", is_error=True)
            content = resolved.read_text()
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)


class WriteFileTool(Tool):
    """Write content to a file in the workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories as needed. Provide a path relative to the workspace root."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    def _resolve(self, path: str) -> Path:
        resolved = (self._workspace_root / path).resolve()
        if not str(resolved).startswith(str(self._workspace_root)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs["path"]
        content = kwargs["content"]
        try:
            resolved = self._resolve(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            return ToolResult(output=f"Successfully wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)


class ListDirectoryTool(Tool):
    """List files and directories in the workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories at a given path relative to the workspace root."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to list. Use '.' for workspace root.",
                    "default": ".",
                },
            },
        }

    def _resolve(self, path: str) -> Path:
        resolved = (self._workspace_root / path).resolve()
        if not str(resolved).startswith(str(self._workspace_root)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", ".")
        try:
            resolved = self._resolve(path)
            if not resolved.exists():
                return ToolResult(output=f"Directory not found: {path}", is_error=True)
            if not resolved.is_dir():
                return ToolResult(output=f"Not a directory: {path}", is_error=True)
            entries = sorted(resolved.iterdir())
            lines = []
            for entry in entries:
                suffix = "/" if entry.is_dir() else ""
                rel = entry.relative_to(self._workspace_root)
                lines.append(f"{rel}{suffix}")
            return ToolResult(output="\n".join(lines) if lines else "(empty directory)")
        except Exception as e:
            return ToolResult(output=str(e), is_error=True)
