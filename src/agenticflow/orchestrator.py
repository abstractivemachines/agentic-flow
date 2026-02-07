"""Orchestrator — Claude-powered dispatch loop that routes work to specialist agents."""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from agenticflow.agents.coder import CoderAgent
from agenticflow.agents.planner import PlannerAgent
from agenticflow.agents.reviewer import ReviewerAgent
from agenticflow.agents.tester import TesterAgent
from agenticflow.models import AgentType, Task, TaskResult, Workspace

logger = logging.getLogger(__name__)

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator of a multi-agent software engineering system. Your job \
is to break down the user's request and dispatch work to specialist agents.

Available agents:
- **planner**: Creates implementation plans. Use for complex tasks that need \
  architecture/design thinking before coding.
- **coder**: Writes and modifies code. Give it clear instructions on what to build.
- **reviewer**: Reviews code for correctness, quality, and security. Use after \
  coding to catch issues.
- **tester**: Writes and runs tests. Use after coding to verify correctness.

Workflow guidelines:
- For non-trivial tasks, start with the planner, then coder, then reviewer/tester.
- For simple tasks, you can go straight to the coder.
- If the reviewer finds issues, dispatch the coder again with the feedback.
- If tests fail, dispatch the coder to fix the issues, then re-test.
- You can dispatch agents multiple times as needed.

Use the dispatch_agent tool to send work to agents. Include all relevant context \
in the task description — agents don't share memory of previous conversations.

When all work is complete, provide a final summary to the user.
"""

DISPATCH_TOOL = {
    "name": "dispatch_agent",
    "description": (
        "Dispatch a task to a specialist agent. The agent will execute the task "
        "and return a result summary."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["planner", "coder", "reviewer", "tester"],
                "description": "Which specialist agent to dispatch.",
            },
            "task_description": {
                "type": "string",
                "description": (
                    "Detailed description of what the agent should do. "
                    "Include all necessary context."
                ),
            },
        },
        "required": ["agent_type", "task_description"],
    },
}

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
MAX_ORCHESTRATOR_TURNS = 20

AGENT_CLASSES = {
    AgentType.PLANNER: PlannerAgent,
    AgentType.CODER: CoderAgent,
    AgentType.REVIEWER: ReviewerAgent,
    AgentType.TESTER: TesterAgent,
}


class Orchestrator:
    """Top-level orchestrator that uses Claude to decide which agents to invoke."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.verbose = verbose
        self.client = anthropic.Anthropic()

    def run(self, user_request: str) -> str:
        """Run the orchestrator loop for a user request. Returns final summary."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_request},
        ]

        for turn in range(MAX_ORCHESTRATOR_TURNS):
            if self.verbose:
                logger.info("[orchestrator] turn %d", turn + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=[DISPATCH_TOOL],
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Orchestrator is done — extract final text
                text_parts = [
                    block.text for block in response.content if hasattr(block, "text")
                ]
                return "\n".join(text_parts) if text_parts else "Done."

            # Process dispatch_agent tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name != "dispatch_agent":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                        "is_error": True,
                    })
                    continue

                result = self._dispatch(block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        return "Orchestrator reached maximum turns without completing."

    def _dispatch(self, tool_input: dict[str, Any]) -> str:
        """Create and run a specialist agent, returning the result as a string."""
        agent_type_str = tool_input["agent_type"]
        task_description = tool_input["task_description"]

        agent_type = AgentType(agent_type_str)
        agent_cls = AGENT_CLASSES[agent_type]

        if self.verbose:
            logger.info("[orchestrator] dispatching to %s", agent_type.value)

        agent = agent_cls(
            workspace=self.workspace,
            model=self.model,
            verbose=self.verbose,
        )

        task = Task(description=task_description, agent_type=agent_type)
        task_result: TaskResult = agent.run(task)

        self.workspace.add_result(task_result)

        # Build a report for the orchestrator
        status = "SUCCESS" if task_result.success else "FAILED"
        report = f"[{status}] {agent_type.value} agent result:\n{task_result.summary}"
        if task_result.errors:
            report += f"\nErrors: {json.dumps(task_result.errors)}"
        return report
