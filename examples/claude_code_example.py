"""Example: using the Claude Code CLI backend.

This backend delegates the agentic loop to Claude Code's built-in tools
(Read, Write, Edit, Bash, etc.) instead of making direct Anthropic API calls.

Prerequisites:
    pip install agenticflow[claude-code]

No ANTHROPIC_API_KEY is needed — the Claude Code CLI handles authentication.
"""

import asyncio
from pathlib import Path

from agenticflow.claude_code import ClaudeCodeOrchestrator
from agenticflow.models import Workspace


async def main() -> None:
    workspace = Workspace(root=Path("./example_workspace"))

    # Optionally specify per-agent models
    orchestrator = ClaudeCodeOrchestrator(
        workspace=workspace,
        model="claude-sonnet-4-5-20250929",
        agent_models={
            "planner": "claude-opus-4-6",
            "coder": "claude-sonnet-4-5-20250929",
        },
        verbose=True,
    )

    result = await orchestrator.run(
        "Create a Python script that reads a CSV file and outputs summary statistics."
    )
    print(result)


async def streaming_example() -> None:
    """Shows how to stream events from the Claude Code backend."""
    workspace = Workspace(root=Path("./example_workspace"))
    orchestrator = ClaudeCodeOrchestrator(workspace=workspace, verbose=True)

    async for event in orchestrator.run_stream("Create a hello world Python script."):
        if event.kind == "text":
            print(event.data, end="", flush=True)
        elif event.kind == "done":
            print("\n--- Done ---")
        elif event.kind == "error":
            print(f"\nError: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
