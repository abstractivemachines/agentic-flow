"""Tool registry — maps agent types to tool sets."""

from __future__ import annotations

from pathlib import Path

from agenticflow.models import AgentType
from agenticflow.tools.base import Tool
from agenticflow.tools.code_exec import RunCommandTool
from agenticflow.tools.file_io import ListDirectoryTool, ReadFileTool, WriteFileTool
from agenticflow.tools.git import GitTool
from agenticflow.tools.http_client import HttpRequestTool
from agenticflow.tools.shell import ShellTool
from agenticflow.tools.web_search import WebSearchTool


def get_tools_for_agent(agent_type: AgentType, workspace_root: Path) -> list[Tool]:
    """Return the tool set appropriate for a given agent type."""
    read = ReadFileTool(workspace_root)
    write = WriteFileTool(workspace_root)
    ls = ListDirectoryTool(workspace_root)
    run = RunCommandTool(workspace_root)
    search = WebSearchTool()
    http = HttpRequestTool()
    git = GitTool(workspace_root)
    shell = ShellTool(workspace_root)

    mapping: dict[AgentType, list[Tool]] = {
        AgentType.PLANNER: [read, ls, search, http],
        AgentType.CODER: [read, write, ls, run, git, shell],
        AgentType.REVIEWER: [read, ls, run],
        AgentType.TESTER: [read, write, ls, run],
    }
    return mapping[agent_type]
