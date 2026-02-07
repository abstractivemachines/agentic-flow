"""CLI entry point for AgenticFlow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agenticflow",
        description="AgenticFlow: Multi-agent software engineering powered by Claude",
    )
    parser.add_argument(
        "task",
        help="Description of the task to perform",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--workspace",
        default="./workspace",
        help="Workspace directory (default: ./workspace)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your API key at https://console.anthropic.com/settings/keys\n"
            "Then run: export ANTHROPIC_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
        )

    workspace = Workspace(root=Path(args.workspace))
    orchestrator = Orchestrator(
        workspace=workspace,
        model=args.model,
        verbose=args.verbose,
    )

    print(f"AgenticFlow — workspace: {workspace.root.resolve()}")
    print(f"Task: {args.task}")
    print("-" * 60)

    try:
        result = orchestrator.run(args.task)
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(result)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
