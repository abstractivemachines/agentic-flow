"""Orchestrator — Claude-powered dispatch loop that routes work to specialist agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Generator

import anthropic

from agenticflow.agents.coder import CoderAgent
from agenticflow.agents.planner import PlannerAgent
from agenticflow.agents.reviewer import ReviewerAgent
from agenticflow.agents.tester import TesterAgent
from agenticflow.models import AgentType, Task, TaskResult, Workspace
from agenticflow.retry import retry_api_call

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

CONTEXT_TOOL = {
    "name": "set_shared_context",
    "description": (
        "Store a value in shared context so subsequent agents can access it. "
        "Use this to pass plans, decisions, or artifacts between agents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key name for the context entry.",
            },
            "value": {
                "type": "string",
                "description": "Value to store.",
            },
        },
        "required": ["key", "value"],
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


@dataclass
class StreamEvent:
    """An event emitted during streaming orchestration."""

    kind: str  # "text", "dispatch", "agent_result", "done", "error"
    data: str


class Orchestrator:
    """Top-level orchestrator that uses Claude to decide which agents to invoke."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model: str = DEFAULT_MODEL,
        agent_models: dict[str, str] | None = None,
        verbose: bool = False,
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.agent_models = agent_models or {}
        self.verbose = verbose
        self.client = anthropic.Anthropic()

    def run(self, user_request: str) -> str:
        """Run the orchestrator loop for a user request. Returns final summary."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_request},
        ]
        tools = [DISPATCH_TOOL, CONTEXT_TOOL]

        for turn in range(MAX_ORCHESTRATOR_TURNS):
            if self.verbose:
                logger.info("[orchestrator] turn %d", turn + 1)

            response = retry_api_call(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text_parts = self._extract_text(response.content)
                return "\n".join(text_parts) if text_parts else "Done."

            tool_results = self._process_tool_calls(response.content)
            messages.append({"role": "user", "content": tool_results})

        return "Orchestrator reached maximum turns without completing."

    def run_stream(self, user_request: str) -> Generator[StreamEvent, None, None]:
        """Run the orchestrator loop, yielding StreamEvents as work progresses."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_request},
        ]
        tools = [DISPATCH_TOOL, CONTEXT_TOOL]

        for turn in range(MAX_ORCHESTRATOR_TURNS):
            if self.verbose:
                logger.info("[orchestrator] turn %d", turn + 1)

            collected_content: list[Any] = []
            stop_reason = None

            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            ) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            if hasattr(event.content_block, "text"):
                                pass  # will accumulate via deltas
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield StreamEvent(kind="text", data=event.delta.text)
                final_message = stream.get_final_message()
                collected_content = final_message.content
                stop_reason = final_message.stop_reason

            messages.append({"role": "assistant", "content": collected_content})

            if stop_reason == "end_turn":
                text_parts = self._extract_text(collected_content)
                summary = "\n".join(text_parts) if text_parts else "Done."
                yield StreamEvent(kind="done", data=summary)
                return

            tool_results = self._process_tool_calls(collected_content)
            for tr in tool_results:
                if not tr.get("is_error"):
                    yield StreamEvent(kind="agent_result", data=tr["content"])
            messages.append({"role": "user", "content": tool_results})

        yield StreamEvent(kind="error", data="Orchestrator reached maximum turns.")

    def _process_tool_calls(self, content: list[Any]) -> list[dict[str, Any]]:
        """Process tool_use blocks and return tool_result dicts."""
        tool_results = []
        for block in content:
            if block.type != "tool_use":
                continue

            if block.name == "dispatch_agent":
                try:
                    result = self._dispatch(block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                except Exception as exc:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"dispatch_agent failed: {exc}",
                        "is_error": True,
                    })
            elif block.name == "set_shared_context":
                try:
                    raw_input = block.input if isinstance(block.input, dict) else {}
                    key = raw_input.get("key", "")
                    value = raw_input.get("value", "")
                    if not isinstance(key, str) or not isinstance(value, str):
                        raise ValueError("set_shared_context requires string key/value")
                    self.workspace.shared_context.set(key, value)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Stored context key '{key}'.",
                    })
                except Exception as exc:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"set_shared_context failed: {exc}",
                        "is_error": True,
                    })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Unknown tool: {block.name}",
                    "is_error": True,
                })
        return tool_results

    @staticmethod
    def _extract_text(content: list[Any]) -> list[str]:
        return [block.text for block in content if hasattr(block, "text")]

    def _dispatch(self, tool_input: dict[str, Any]) -> str:
        """Create and run a specialist agent, returning the result as a string."""
        if not isinstance(tool_input, dict):
            raise ValueError("dispatch_agent input must be an object")

        agent_type_str = tool_input.get("agent_type")
        task_description = tool_input.get("task_description")
        if not isinstance(agent_type_str, str):
            raise ValueError("dispatch_agent requires string 'agent_type'")
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("dispatch_agent requires non-empty 'task_description'")

        agent_type = AgentType(agent_type_str)
        agent_cls = AGENT_CLASSES[agent_type]

        if self.verbose:
            logger.info("[orchestrator] dispatching to %s", agent_type.value)

        agent_model = self.agent_models.get(agent_type.value, self.model)

        agent = agent_cls(
            workspace=self.workspace,
            model=agent_model,
            verbose=self.verbose,
        )

        task = Task(description=task_description, agent_type=agent_type)
        task_result: TaskResult = agent.run(task)

        self.workspace.add_result(task_result)

        # Store the result summary in shared context
        ctx_key = f"result_{agent_type.value}_{task.id}"
        self.workspace.shared_context.set(ctx_key, task_result.summary)

        status = "SUCCESS" if task_result.success else "FAILED"
        report = f"[{status}] {agent_type.value} agent result:\n{task_result.summary}"
        if task_result.errors:
            report += f"\nErrors: {json.dumps(task_result.errors)}"
        return report
