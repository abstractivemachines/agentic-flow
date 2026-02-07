"""Example: using different models for different agents."""

from pathlib import Path

from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator


def main() -> None:
    workspace = Workspace(root=Path("./example_workspace"))

    orchestrator = Orchestrator(
        workspace=workspace,
        model="claude-sonnet-4-5-20250929",  # default for orchestrator
        agent_models={
            "planner": "claude-opus-4-6",           # use Opus for planning
            "coder": "claude-sonnet-4-5-20250929",  # Sonnet for coding
            "reviewer": "claude-opus-4-6",          # Opus for reviews
            "tester": "claude-sonnet-4-5-20250929",  # Sonnet for tests
        },
        verbose=True,
    )

    result = orchestrator.run(
        "Design and implement a URL shortener service with a REST API."
    )
    print(result)


if __name__ == "__main__":
    main()
