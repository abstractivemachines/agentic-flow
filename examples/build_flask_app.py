#!/usr/bin/env python3
"""Example: use AgenticFlow to build a simple Flask app."""

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

    result = orchestrator.run(
        "Build a simple Flask web application with a single endpoint GET /health "
        "that returns {'status': 'ok'}. Include a requirements.txt file."
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    main()
