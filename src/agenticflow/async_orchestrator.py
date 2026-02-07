"""Async orchestrator — runs the dispatch loop using AsyncAnthropic."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import anthropic

from agenticflow.models import AgentType, Task, TaskResult, Workspace
from agenticflow.orchestrator import (
    AGENT_CLASSES,
    CONTEXT_TOOL,
    DEFAULT_MODEL,
    DISPATCH_TOOL,
    MAX_ORCHESTRATOR_TURNS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    StreamEvent,
)
from agenticflow.retry import async_retry_api_call

logger = logging.getLogger(__name__)


class AsyncOrchestrator:
    """Async version of the orchestrator using AsyncAnthropic."""

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
        self.client = anthropic.AsyncAnthropic()

    async def run(self, user_request: str) -> str:
        """Run the orchestrator loop asynchronously. Returns final summary."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_request},
        ]
        tools = [DISPATCH_TOOL, CONTEXT_TOOL]

        for turn in range(MAX_ORCHESTRATOR_TURNS):
            if self.verbose:
                logger.info("[async-orchestrator] turn %d", turn + 1)

            response = await async_retry_api_call(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text_parts = [
                    block.text for block in response.content if hasattr(block, "text")
                ]
                return "\n".join(text_parts) if text_parts else "Done."

            tool_results = self._process_tool_calls(response.content)
            messages.append({"role": "user", "content": tool_results})

        return "Orchestrator reached maximum turns without completing."

    async def run_stream(
        self, user_request: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming orchestrator loop."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_request},
        ]
        tools = [DISPATCH_TOOL, CONTEXT_TOOL]

        for turn in range(MAX_ORCHESTRATOR_TURNS):
            if self.verbose:
                logger.info("[async-orchestrator] turn %d", turn + 1)

            collected_content: list[Any] = []
            stop_reason = None

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield StreamEvent(kind="text", data=event.delta.text)
                final_message = await stream.get_final_message()
                collected_content = final_message.content
                stop_reason = final_message.stop_reason

            messages.append({"role": "assistant", "content": collected_content})

            if stop_reason == "end_turn":
                yield StreamEvent(kind="done", data="")
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
                result = self._dispatch(block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            elif block.name == "set_shared_context":
                key = block.input.get("key", "")
                value = block.input.get("value", "")
                self.workspace.shared_context.set(key, value)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Stored context key '{key}'.",
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Unknown tool: {block.name}",
                    "is_error": True,
                })
        return tool_results

    def _dispatch(self, tool_input: dict[str, Any]) -> str:
        """Create and run a specialist agent, returning the result as a string.

        Note: agent execution is synchronous since agents use the sync Anthropic client.
        """
        agent_type_str = tool_input["agent_type"]
        task_description = tool_input["task_description"]

        agent_type = AgentType(agent_type_str)
        agent_cls = AGENT_CLASSES[agent_type]

        if self.verbose:
            logger.info("[async-orchestrator] dispatching to %s", agent_type.value)

        agent_model = self.agent_models.get(agent_type.value, self.model)

        agent = agent_cls(
            workspace=self.workspace,
            model=agent_model,
            verbose=self.verbose,
        )

        task = Task(description=task_description, agent_type=agent_type)
        task_result: TaskResult = agent.run(task)

        self.workspace.add_result(task_result)

        ctx_key = f"result_{agent_type.value}_{task.id}"
        self.workspace.shared_context.set(ctx_key, task_result.summary)

        status = "SUCCESS" if task_result.success else "FAILED"
        report = f"[{status}] {agent_type.value} agent result:\n{task_result.summary}"
        if task_result.errors:
            report += f"\nErrors: {json.dumps(task_result.errors)}"
        return report
