"""CLI entry point for AgenticFlow."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
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
        nargs="?",
        default=None,
        help="Description of the task to perform (optional when resuming with --backend langgraph --thread-id)",
    )
    parser.add_argument(
        "--backend",
        choices=["api", "claude-code", "langgraph"],
        default="api",
        help="Backend to use (default: api)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model to use (default: claude-sonnet-4-5-20250929 for api backend)",
    )
    parser.add_argument(
        "--workspace",
        default="./workspace",
        help="Workspace directory (default: ./workspace)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    # LangGraph-specific arguments
    parser.add_argument(
        "--checkpoint-backend",
        choices=["sqlite", "memory"],
        default="sqlite",
        help="Checkpoint backend for langgraph (default: sqlite)",
    )
    parser.add_argument(
        "--approval-policy",
        choices=["none", "default", "strict"],
        default="none",
        help="HITL approval policy for langgraph (default: none)",
    )
    parser.add_argument(
        "--no-approve",
        action="store_true",
        help="Shorthand for --approval-policy none",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread ID for langgraph (used for resume; can also be provided for new runs)",
    )

    args = parser.parse_args(argv)

    if args.no_approve:
        args.approval_policy = "none"

    if args.task is None and not (args.backend == "langgraph" and args.thread_id):
        parser.error(
            "task is required unless resuming a langgraph run with --backend langgraph --thread-id"
        )

    if args.backend in ("api", "langgraph") and not os.environ.get("ANTHROPIC_API_KEY"):
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

    print(
        f"AgenticFlow — backend: {args.backend}, workspace: {workspace.root.resolve()}"
    )
    if args.task is not None:
        print(f"Task: {args.task}")
    elif args.thread_id:
        print(f"Resuming thread: {args.thread_id}")
    print("-" * 60)

    try:
        if args.backend == "claude-code":
            from agenticflow.claude_code import ClaudeCodeOrchestrator

            assert args.task is not None
            orchestrator = ClaudeCodeOrchestrator(
                workspace=workspace,
                model=args.model,
                verbose=args.verbose,
                invocation_dir=Path.cwd(),
            )
            result = asyncio.run(orchestrator.run(args.task))
        elif args.backend == "langgraph":
            from agenticflow.langgraph_orchestrator import LangGraphOrchestrator

            model = args.model or "claude-sonnet-4-5-20250929"
            lg_orchestrator = LangGraphOrchestrator(
                workspace=workspace,
                model=model,
                checkpoint_backend=args.checkpoint_backend,
                approval_policy=args.approval_policy,
                verbose=args.verbose,
            )
            if args.task is None:
                if args.thread_id is None:
                    parser.error(
                        "--thread-id is required when resuming a langgraph run without a task"
                    )
                result = lg_orchestrator.resume(args.thread_id)
            else:
                run_thread_id = args.thread_id or uuid.uuid4().hex[:12]
                print(f"Thread ID: {run_thread_id}")
                result = lg_orchestrator.run(args.task, thread_id=run_thread_id)
        else:
            model = args.model or "claude-sonnet-4-5-20250929"
            orchestrator = Orchestrator(
                workspace=workspace,
                model=model,
                verbose=args.verbose,
            )
            assert args.task is not None
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
