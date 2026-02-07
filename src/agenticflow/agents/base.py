"""Base agent with the core agentic loop."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from agenticflow.models import AgentType, Task, TaskResult, Workspace
from agenticflow.tools import get_tools_for_agent
from agenticflow.tools.base import Tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096
MAX_TURNS = 25


class Agent:
    """Base agent that runs an agentic tool-use loop against the Claude API."""

    agent_type: AgentType
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = DEFAULT_MAX_TOKENS

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
        self.tools: list[Tool] = get_tools_for_agent(self.agent_type, workspace.root)
        self._tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

    def run(self, task: Task) -> TaskResult:
        """Execute the agentic loop for the given task."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.to_prompt()},
        ]
        tool_defs = [t.to_anthropic_dict() for t in self.tools]

        for turn in range(MAX_TURNS):
            if self.verbose:
                logger.info("[%s] turn %d", self.agent_type.value, turn + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=tool_defs,
                messages=messages,
            )

            # Collect the assistant message
            messages.append({"role": "assistant", "content": response.content})

            # If the model stopped without requesting tools, we're done
            if response.stop_reason == "end_turn":
                return self._build_result(task, response)

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool = self._tool_map.get(block.name)
                if tool is None:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                        "is_error": True,
                    })
                    continue

                if self.verbose:
                    logger.info("[%s] calling %s(%s)", self.agent_type.value, block.name, block.input)

                result = tool.execute(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.output,
                    "is_error": result.is_error,
                })

            messages.append({"role": "user", "content": tool_results})

        # Exhausted turns
        return TaskResult(
            task_id=task.id,
            agent_type=self.agent_type,
            success=False,
            summary=f"Agent exhausted {MAX_TURNS} turns without completing.",
            errors=[f"Max turns ({MAX_TURNS}) reached"],
        )

    def _build_result(self, task: Task, response: Any) -> TaskResult:
        """Extract final text from the response and build a TaskResult."""
        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        summary = "\n".join(text_parts) if text_parts else "Task completed (no text output)."
        return TaskResult(
            task_id=task.id,
            agent_type=self.agent_type,
            success=True,
            summary=summary,
        )
