"""Claude Code CLI backend — uses claude-agent-sdk instead of direct API calls."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, AsyncGenerator

from agenticflow.models import AgentType, Task, TaskResult, Workspace
from agenticflow.orchestrator import (
    MAX_ORCHESTRATOR_TURNS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    StreamEvent,
)

logger = logging.getLogger(__name__)


async def _streaming_prompt(text: str) -> AsyncIterator[dict[str, Any]]:
    """Wrap a string prompt as an AsyncIterable in the SDK's streaming format.

    When MCP servers are present, the SDK must receive the prompt as an
    AsyncIterable so that stdin stays open for bidirectional control protocol
    messages (MCP tool call/response). A plain string prompt causes the SDK
    to call ``end_input()`` immediately, closing stdin before MCP responses
    can be written back.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }


def _import_sdk() -> Any:
    """Lazily import claude-agent-sdk, raising a clear error if not installed."""
    try:
        import claude_agent_sdk
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for the Claude Code backend.\n"
            "Install it with: pip install agenticflow[claude-code]\n"
            "Or directly: pip install claude-agent-sdk"
        ) from None
    return claude_agent_sdk


# Maps AgentType to the Claude Code built-in tool names the agent is allowed to use.
CLAUDE_CODE_TOOL_MAPPING: dict[AgentType, list[str]] = {
    AgentType.PLANNER: ["Read", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"],
    AgentType.CODER: ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
    AgentType.REVIEWER: ["Read", "Glob", "Grep", "Bash"],
    AgentType.TESTER: ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
}

_TOOL_NOTE = (
    "\n\nNote: You have access to Claude Code native tools (Read, Write, Edit, "
    "Bash, Glob, Grep, etc.) rather than the AgenticFlow tool names mentioned "
    "above (read_file, write_file, etc.). Use the Claude Code tools directly."
)


def _build_agent_system_prompt(
    agent_type: AgentType, workspace: Workspace
) -> dict[str, Any]:
    """Build a SystemPromptPreset dict for the given agent type.

    Uses Claude Code's default system prompt as the base, appending the
    agent-specific instructions and any shared context.
    """
    from agenticflow.agents.coder import CoderAgent
    from agenticflow.agents.planner import PlannerAgent
    from agenticflow.agents.reviewer import ReviewerAgent
    from agenticflow.agents.tester import TesterAgent

    agent_prompts = {
        AgentType.PLANNER: PlannerAgent.system_prompt,
        AgentType.CODER: CoderAgent.system_prompt,
        AgentType.REVIEWER: ReviewerAgent.system_prompt,
        AgentType.TESTER: TesterAgent.system_prompt,
    }

    parts = [agent_prompts[agent_type], _TOOL_NOTE]
    ctx_fragment = workspace.shared_context.to_prompt_fragment()
    if ctx_fragment:
        parts.append(ctx_fragment)

    return {
        "type": "preset",
        "preset": "claude_code",
        "append": "\n\n".join(parts),
    }


async def run_agent(
    agent_type: AgentType,
    task_description: str,
    workspace: Workspace,
    *,
    model: str | None = None,
    verbose: bool = False,
    add_dirs: list[str] | None = None,
) -> TaskResult:
    """Run a specialist agent via Claude Code CLI.

    Calls ``query()`` with the agent's system prompt and allowed tools.
    Returns a :class:`TaskResult` with the final result or error.
    """
    sdk = _import_sdk()

    task = Task(description=task_description, agent_type=agent_type)
    system_prompt = _build_agent_system_prompt(agent_type, workspace)
    allowed_tools = CLAUDE_CODE_TOOL_MAPPING[agent_type]

    options = sdk.ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        cwd=str(workspace.root.resolve()),
        permission_mode="acceptEdits",
        max_turns=25,
        **({"add_dirs": add_dirs} if add_dirs else {}),
    )
    if model:
        options.model = model

    try:
        result_text = None
        async for message in sdk.query(prompt=task_description, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        result_text = block.text
                        if verbose:
                            logger.info(
                                "[%s] text: %s", agent_type.value, block.text[:200]
                            )
            elif isinstance(message, sdk.ResultMessage):
                if message.result:
                    result_text = message.result
                if message.is_error:
                    return TaskResult(
                        task_id=task.id,
                        agent_type=agent_type,
                        success=False,
                        summary=result_text or "Agent returned an error.",
                        errors=[result_text or "Unknown error"],
                    )

        return TaskResult(
            task_id=task.id,
            agent_type=agent_type,
            success=True,
            summary=result_text or "Task completed (no text output).",
        )
    except Exception as exc:
        return TaskResult(
            task_id=task.id,
            agent_type=agent_type,
            success=False,
            summary=f"Agent error: {exc}",
            errors=[str(exc)],
        )


def _build_orchestrator_mcp_server(
    workspace: Workspace,
    agent_models: dict[str, str],
    default_model: str | None,
    verbose: bool,
    add_dirs: list[str] | None = None,
) -> Any:
    """Build an in-process MCP server with dispatch_agent and set_shared_context tools."""
    sdk = _import_sdk()

    @sdk.tool(
        "dispatch_agent",
        "Dispatch a task to a specialist agent. The agent will execute the task "
        "and return a result summary.",
        {
            "agent_type": str,
            "task_description": str,
        },
    )
    async def dispatch_agent(args: dict[str, Any]) -> dict[str, Any]:
        agent_type_str = args["agent_type"]
        task_description = args["task_description"]

        agent_type = AgentType(agent_type_str)
        agent_model = agent_models.get(agent_type.value, default_model)

        if verbose:
            logger.info(
                "[claude-code-orchestrator] dispatching to %s", agent_type.value
            )

        task_result = await run_agent(
            agent_type,
            task_description,
            workspace,
            model=agent_model,
            verbose=verbose,
            add_dirs=add_dirs,
        )

        workspace.add_result(task_result)

        # Store the result summary in shared context
        ctx_key = f"result_{agent_type.value}_{task_result.task_id}"
        workspace.shared_context.set(ctx_key, task_result.summary)

        status = "SUCCESS" if task_result.success else "FAILED"
        report = f"[{status}] {agent_type.value} agent result:\n{task_result.summary}"
        if task_result.errors:
            report += f"\nErrors: {json.dumps(task_result.errors)}"

        return {
            "content": [{"type": "text", "text": report}],
        }

    @sdk.tool(
        "set_shared_context",
        "Store a value in shared context so subsequent agents can access it. "
        "Use this to pass plans, decisions, or artifacts between agents.",
        {
            "key": str,
            "value": str,
        },
    )
    async def set_shared_context(args: dict[str, Any]) -> dict[str, Any]:
        key = args["key"]
        value = args["value"]
        workspace.shared_context.set(key, value)
        return {
            "content": [{"type": "text", "text": f"Stored context key '{key}'."}],
        }

    return sdk.create_sdk_mcp_server(
        name="agenticflow",
        tools=[dispatch_agent, set_shared_context],
    )


class ClaudeCodeOrchestrator:
    """Orchestrator that uses the Claude Code CLI via claude-agent-sdk.

    Unlike the standard :class:`Orchestrator`, this does not require an
    ``ANTHROPIC_API_KEY`` — the Claude Code CLI handles authentication.
    The agentic loop and tool execution are delegated to Claude Code's
    built-in tools (Read, Write, Edit, Bash, etc.).

    Note: ``permission_mode="acceptEdits"`` is used, which auto-approves
    tool calls. For strict sandboxing, use the ``api`` backend or run
    in Docker.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        model: str | None = None,
        agent_models: dict[str, str] | None = None,
        verbose: bool = False,
        invocation_dir: Path | None = None,
    ) -> None:
        _import_sdk()  # Validate SDK is available early
        self.workspace = workspace
        self.model = model
        self.agent_models = agent_models or {}
        self.verbose = verbose
        self.invocation_dir = invocation_dir

    def _build_add_dirs(self) -> list[str] | None:
        """Build the add_dirs list from invocation_dir if it differs from workspace root."""
        if self.invocation_dir is None:
            return None
        inv = self.invocation_dir.resolve()
        ws = self.workspace.root.resolve()
        if inv == ws:
            return None
        return [str(inv)]

    async def run(self, user_request: str) -> str:
        """Run the orchestrator loop via Claude Code CLI. Returns final summary."""
        sdk = _import_sdk()
        add_dirs = self._build_add_dirs()

        server = _build_orchestrator_mcp_server(
            self.workspace,
            self.agent_models,
            self.model,
            self.verbose,
            add_dirs=add_dirs,
        )

        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": ORCHESTRATOR_SYSTEM_PROMPT,
        }

        options = sdk.ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"agenticflow": server},
            allowed_tools=[
                "mcp__agenticflow__dispatch_agent",
                "mcp__agenticflow__set_shared_context",
            ],
            cwd=str(self.workspace.root.resolve()),
            permission_mode="acceptEdits",
            max_turns=MAX_ORCHESTRATOR_TURNS,
            **({"add_dirs": add_dirs} if add_dirs else {}),
        )
        if self.model:
            options.model = self.model

        result_text = None
        prompt = _streaming_prompt(user_request)
        async for message in sdk.query(prompt=prompt, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        result_text = block.text
            elif isinstance(message, sdk.ResultMessage):
                if message.result:
                    result_text = message.result

        return result_text or "Done."

    async def run_stream(self, user_request: str) -> AsyncGenerator[StreamEvent, None]:
        """Run the orchestrator loop, yielding StreamEvents as work progresses."""
        sdk = _import_sdk()
        add_dirs = self._build_add_dirs()

        server = _build_orchestrator_mcp_server(
            self.workspace,
            self.agent_models,
            self.model,
            self.verbose,
            add_dirs=add_dirs,
        )

        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": ORCHESTRATOR_SYSTEM_PROMPT,
        }

        options = sdk.ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"agenticflow": server},
            allowed_tools=[
                "mcp__agenticflow__dispatch_agent",
                "mcp__agenticflow__set_shared_context",
            ],
            cwd=str(self.workspace.root.resolve()),
            permission_mode="acceptEdits",
            max_turns=MAX_ORCHESTRATOR_TURNS,
            include_partial_messages=True,
            **({"add_dirs": add_dirs} if add_dirs else {}),
        )
        if self.model:
            options.model = self.model

        try:
            prompt = _streaming_prompt(user_request)
            async for message in sdk.query(prompt=prompt, options=options):
                if hasattr(message, "event"):
                    # StreamEvent from the SDK
                    event = message.event
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield StreamEvent(kind="text", data=delta.get("text", ""))
                elif isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            yield StreamEvent(kind="text", data=block.text)
                elif isinstance(message, sdk.ResultMessage):
                    if message.is_error:
                        yield StreamEvent(
                            kind="error",
                            data=message.result or "Orchestrator error.",
                        )
                    else:
                        yield StreamEvent(kind="done", data="")
                    return
        except Exception as exc:
            yield StreamEvent(kind="error", data=str(exc))

    def run_sync(self, user_request: str) -> str:
        """Convenience synchronous wrapper around :meth:`run`."""
        return asyncio.run(self.run(user_request))
