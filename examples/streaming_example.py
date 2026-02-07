"""Example: streaming orchestrator output."""

from pathlib import Path

from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator


def main() -> None:
    workspace = Workspace(root=Path("./example_workspace"))
    orchestrator = Orchestrator(
        workspace=workspace,
        model="claude-sonnet-4-5-20250929",
        verbose=True,
    )

    for event in orchestrator.run_stream("Write a Python function that sorts a list using merge sort"):
        if event.kind == "text":
            print(event.data, end="", flush=True)
        elif event.kind == "agent_result":
            print(f"\n--- Agent Result ---\n{event.data}\n")
        elif event.kind == "done":
            print("\n--- Done ---")
        elif event.kind == "error":
            print(f"\n--- Error: {event.data} ---")


if __name__ == "__main__":
    main()
