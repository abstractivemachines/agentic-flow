"""Example: manually using shared context between agents."""

from pathlib import Path

from agenticflow.agents.coder import CoderAgent
from agenticflow.agents.planner import PlannerAgent
from agenticflow.models import AgentType, Task, Workspace


def main() -> None:
    workspace = Workspace(root=Path("./example_workspace"))

    # Store some context that agents will see
    workspace.shared_context.set("project_language", "Python 3.12")
    workspace.shared_context.set("style_guide", "Use type hints everywhere, follow PEP 8")

    # Run planner — it sees the shared context in its system prompt
    planner = PlannerAgent(workspace=workspace, verbose=True)
    plan_result = planner.run(
        Task(description="Plan a CLI calculator app", agent_type=AgentType.PLANNER)
    )
    print(f"Plan: {plan_result.summary}")

    # Store the plan so the coder can see it
    workspace.shared_context.set("implementation_plan", plan_result.summary)

    # Run coder — it sees the plan and style guide in its system prompt
    coder = CoderAgent(workspace=workspace, verbose=True)
    code_result = coder.run(
        Task(description="Implement the calculator app per the plan", agent_type=AgentType.CODER)
    )
    print(f"Code: {code_result.summary}")


if __name__ == "__main__":
    main()
