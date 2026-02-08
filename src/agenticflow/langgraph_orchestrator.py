"""LangGraph-based orchestrator with checkpointing, HITL, and Studio support."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Generator

import anthropic

from agenticflow.langgraph_state import (
    OrchestratorState,
    serialize_task_result,
)
from agenticflow.models import AgentType, Task, Workspace
from agenticflow.orchestrator import AGENT_CLASSES, DEFAULT_MODEL, StreamEvent
from agenticflow.retry import retry_api_call

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_langgraph() -> Any:
    """Lazily import langgraph, raising a clear error if not installed."""
    try:
        import langgraph  # noqa: F811
    except ImportError:
        raise ImportError(
            "langgraph is required for the LangGraph backend.\n"
            "Install it with: pip install agenticflow[langgraph]\n"
            "Or directly: pip install langgraph langgraph-checkpoint-sqlite"
        ) from None
    return langgraph


# ---------------------------------------------------------------------------
# Workspace registry — nodes look up the workspace by path from config
# ---------------------------------------------------------------------------

_WORKSPACE_REGISTRY: dict[str, Workspace] = {}


def _register_workspace(workspace: Workspace) -> str:
    key = str(workspace.root.resolve())
    _WORKSPACE_REGISTRY[key] = workspace
    return key


def _get_workspace(config: Any) -> Workspace:
    key = config.get("configurable", {}).get("workspace_key", "")
    ws = _WORKSPACE_REGISTRY.get(key)
    if ws is None:
        raise RuntimeError(f"Workspace not found in registry for key: {key}")
    return ws


# ---------------------------------------------------------------------------
# Approval policies
# ---------------------------------------------------------------------------

ApprovalPolicy = Callable[[str], bool]


def default_approval_policy(agent_type: str) -> bool:
    """Gate only coder agents (they write files)."""
    return agent_type == "coder"


def strict_approval_policy(agent_type: str) -> bool:
    """Gate all agents."""
    return True


def no_approval_policy(agent_type: str) -> bool:
    """Never gate."""
    return False


APPROVAL_POLICIES: dict[str, ApprovalPolicy] = {
    "default": default_approval_policy,
    "strict": strict_approval_policy,
    "none": no_approval_policy,
}

# ---------------------------------------------------------------------------
# Planning prompt
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are a planning assistant for a multi-agent software engineering system.
Given a user request, decompose it into an ordered list of agent steps.

Available agent types:
- planner: Architecture and design thinking
- coder: Writing and modifying code
- reviewer: Code review for correctness, quality, security
- tester: Writing and running tests

Output ONLY a JSON array of objects. Each object must have:
- "agent_type": one of "planner", "coder", "reviewer", "tester"
- "task_description": detailed description of what the agent should do

Example output:
[
  {"agent_type": "planner", "task_description": "Design the architecture for ..."},
  {"agent_type": "coder", "task_description": "Implement ... based on the plan"},
  {"agent_type": "reviewer", "task_description": "Review the implementation for ..."},
  {"agent_type": "tester", "task_description": "Write tests for ..."}
]

Guidelines:
- For simple tasks, 1-2 steps (e.g. just coder, or coder + tester).
- For complex tasks, start with planner, then coder, then reviewer/tester.
- Include all relevant context in each task_description — agents don't share memory.
- Output ONLY the JSON array, no markdown fences or extra text.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def plan_node(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Call Claude to decompose the user request into an ordered plan."""
    model = config.get("configurable", {}).get("model", DEFAULT_MODEL)
    client = anthropic.Anthropic()

    response = retry_api_call(
        client.messages.create,
        model=model,
        max_tokens=4096,
        system=PLANNING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": state["user_request"]}],
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    try:
        plan = json.loads(text)
        if not isinstance(plan, list):
            raise ValueError("Plan must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "error": f"Failed to parse plan: {exc}\nRaw output: {text[:500]}",
            "is_complete": True,
        }

    return {
        "plan": plan,
        "current_step_index": 0,
        "agent_results": [],
        "shared_context": {},
    }


def should_approve_node(
    state: OrchestratorState, config: RunnableConfig
) -> dict[str, Any]:
    """Check if the current step needs approval; if so, interrupt."""
    from langgraph.types import interrupt

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if idx >= len(plan):
        return {"pending_approval": False}

    step = plan[idx]
    agent_type = step.get("agent_type", "")

    policy_name = config.get("configurable", {}).get("approval_policy", "none")
    policy = APPROVAL_POLICIES.get(policy_name, no_approval_policy)

    if policy(agent_type):
        # Interrupt for human approval
        decision = interrupt(
            {
                "question": f"Approve dispatching {agent_type} agent?",
                "step": step,
                "step_index": idx,
            }
        )
        # After resume, decision is the value passed via Command(resume=...)
        if isinstance(decision, dict):
            return {
                "pending_approval": False,
                "approval_decision": decision.get("decision", "approve"),
                "approval_feedback": decision.get("feedback"),
            }
        return {
            "pending_approval": False,
            "approval_decision": "approve",
            "approval_feedback": None,
        }

    return {
        "pending_approval": False,
        "approval_decision": "approve",
        "approval_feedback": None,
    }


def execute_agent_node(
    state: OrchestratorState, config: RunnableConfig
) -> dict[str, Any]:
    """Run the current agent step using existing Agent classes."""
    workspace = _get_workspace(config)
    model = config.get("configurable", {}).get("model", DEFAULT_MODEL)
    verbose = config.get("configurable", {}).get("verbose", False)

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    step = plan[idx]

    agent_type = AgentType(step["agent_type"])
    agent_cls = AGENT_CLASSES[agent_type]

    agent = agent_cls(workspace=workspace, model=model, verbose=verbose)

    # Include shared context in the task description
    shared_ctx = state.get("shared_context", {})
    task_desc = step["task_description"]
    if shared_ctx:
        ctx_lines = ["Previous context:"]
        for k, v in shared_ctx.items():
            text = str(v)
            if len(text) > 2000:
                text = text[:2000] + "... (truncated)"
            ctx_lines.append(f"  [{k}]: {text}")
        task_desc = task_desc + "\n\n" + "\n".join(ctx_lines)

    task = Task(description=task_desc, agent_type=agent_type)
    result = agent.run(task)
    workspace.add_result(result)

    # Update state
    results = list(state.get("agent_results", []))
    results.append(serialize_task_result(result))

    ctx = dict(shared_ctx)
    ctx_key = f"result_{agent_type.value}_{task.id}"
    ctx[ctx_key] = result.summary

    return {
        "agent_results": results,
        "shared_context": ctx,
        "needs_revision": False,
        "revision_feedback": None,
    }


def evaluate_node(state: OrchestratorState) -> dict[str, Any]:
    """Check the latest result and decide whether to revise, continue, or finalize."""
    results = state.get("agent_results", [])
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if not results:
        return {"is_complete": True, "error": "No results to evaluate"}

    latest = results[-1]

    # If the agent failed and it was a coder, attempt revision
    if not latest.get("success", True) and plan[idx].get("agent_type") == "coder":
        return {
            "needs_revision": True,
            "revision_feedback": f"Agent failed: {latest.get('summary', 'unknown error')}",
        }

    # If a reviewer returned REQUEST_CHANGES, revise
    if plan[idx].get("agent_type") == "reviewer" and "REQUEST_CHANGES" in latest.get(
        "summary", ""
    ):
        return {
            "needs_revision": True,
            "revision_feedback": latest.get("summary", ""),
            # Back up index to re-run coder (the step before reviewer)
            "current_step_index": max(0, idx - 1),
        }

    # Move to next step
    next_idx = idx + 1
    if next_idx >= len(plan):
        return {"current_step_index": next_idx, "is_complete": True}

    return {"current_step_index": next_idx}


def revise_node(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Re-dispatch coder with revision feedback."""
    workspace = _get_workspace(config)
    model = config.get("configurable", {}).get("model", DEFAULT_MODEL)
    verbose = config.get("configurable", {}).get("verbose", False)

    feedback = state.get("revision_feedback", "Please fix the issues.")
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    # Find the original task description from the step
    original_desc = plan[idx].get("task_description", "") if idx < len(plan) else ""

    task_desc = (
        f"REVISION REQUIRED:\n{feedback}\n\n"
        f"Original task: {original_desc}\n\n"
        f"Please fix the issues identified above."
    )

    agent_cls = AGENT_CLASSES[AgentType.CODER]
    agent = agent_cls(workspace=workspace, model=model, verbose=verbose)

    task = Task(description=task_desc, agent_type=AgentType.CODER)
    result = agent.run(task)
    workspace.add_result(result)

    results = list(state.get("agent_results", []))
    results.append(serialize_task_result(result))

    return {
        "agent_results": results,
        "needs_revision": False,
        "revision_feedback": None,
    }


def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """Aggregate results into a final summary."""
    results = state.get("agent_results", [])
    error = state.get("error")

    if error:
        return {"final_summary": f"Error: {error}", "is_complete": True}

    # Check for rejection (may happen before any agents run)
    decision = state.get("approval_decision")
    if decision == "reject":
        feedback = state.get("approval_feedback", "")
        lines = ["Execution was rejected by user."]
        if feedback:
            lines.append(f"Feedback: {feedback}")
        if results:
            lines.append("\nPartial results:")
            for i, r in enumerate(results, 1):
                status = "SUCCESS" if r.get("success") else "FAILED"
                agent = r.get("agent_type", "unknown")
                lines.append(f"  {i}. [{status}] {agent}: {r.get('summary', '')}")
        return {"final_summary": "\n".join(lines), "is_complete": True}

    if not results:
        return {"final_summary": "No work was performed.", "is_complete": True}

    lines = ["Orchestration complete. Results:\n"]
    for i, r in enumerate(results, 1):
        status = "SUCCESS" if r.get("success") else "FAILED"
        agent = r.get("agent_type", "unknown")
        summary = r.get("summary", "")
        lines.append(f"  {i}. [{status}] {agent}: {summary}")

    return {"final_summary": "\n".join(lines), "is_complete": True}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_plan(state: OrchestratorState) -> str:
    """Route based on whether planning succeeded."""
    if state.get("is_complete") or state.get("error"):
        return "finalize"
    return "should_approve"


def route_after_approval(state: OrchestratorState) -> str:
    """Route based on approval decision."""
    decision = state.get("approval_decision", "approve")
    if decision == "reject":
        return "finalize"
    return "execute_agent"


def route_after_evaluate(state: OrchestratorState) -> str:
    """Route based on evaluation of agent result."""
    if state.get("needs_revision"):
        return "revise"
    if state.get("is_complete"):
        return "finalize"
    # More steps to go — loop back to approval check
    return "should_approve"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct and return the StateGraph (uncompiled)."""
    _import_langgraph()
    from langgraph.graph import END, START, StateGraph

    from agenticflow.langgraph_state import OrchestratorState

    builder = StateGraph(OrchestratorState)

    builder.add_node("plan", plan_node)
    builder.add_node("should_approve", should_approve_node)
    builder.add_node("execute_agent", execute_agent_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("revise", revise_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {"should_approve": "should_approve", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "should_approve",
        route_after_approval,
        {"execute_agent": "execute_agent", "finalize": "finalize"},
    )
    builder.add_edge("execute_agent", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "revise": "revise",
            "finalize": "finalize",
            "should_approve": "should_approve",
        },
    )
    builder.add_edge("revise", "evaluate")
    builder.add_edge("finalize", END)

    return builder


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------


class LangGraphOrchestrator:
    """Orchestrator using LangGraph for checkpointing, HITL, and Studio debugging."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model: str = DEFAULT_MODEL,
        checkpoint_backend: str = "sqlite",
        approval_policy: str = "none",
        verbose: bool = False,
    ) -> None:
        _import_langgraph()

        self.workspace = workspace
        self.model = model
        self.approval_policy = approval_policy
        self.verbose = verbose
        self._workspace_key = _register_workspace(workspace)

        self._checkpointer = self._create_checkpointer(checkpoint_backend)
        self._graph = build_graph().compile(checkpointer=self._checkpointer)

    def _create_checkpointer(self, backend: str) -> Any:
        if backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        elif backend == "sqlite":
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            db_dir = self.workspace.root / ".agenticflow"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "checkpoints.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            return SqliteSaver(conn)
        else:
            raise ValueError(f"Unknown checkpoint backend: {backend}")

    def _make_config(self, thread_id: str | None = None) -> dict[str, Any]:
        tid = thread_id or uuid.uuid4().hex[:12]
        return {
            "configurable": {
                "thread_id": tid,
                "workspace_key": self._workspace_key,
                "model": self.model,
                "approval_policy": self.approval_policy,
                "verbose": self.verbose,
            }
        }

    def run(self, user_request: str, *, thread_id: str | None = None) -> str:
        """Run the graph to completion, handling HITL interrupts via stdin."""
        config = self._make_config(thread_id)
        initial_state: OrchestratorState = {
            "user_request": user_request,
            "plan": [],
            "current_step_index": 0,
            "agent_results": [],
            "shared_context": {},
            "pending_approval": False,
            "approval_decision": None,
            "approval_feedback": None,
            "needs_revision": False,
            "revision_feedback": None,
            "final_summary": None,
            "is_complete": False,
            "error": None,
        }

        result_state = self._graph.invoke(initial_state, config)

        # Handle interrupt loop
        while True:
            graph_state = self._graph.get_state(config)
            if not graph_state.tasks:
                break

            # There's an interrupt — prompt the user
            task_data = graph_state.tasks[0]
            interrupt_value = getattr(task_data, "interrupts", [{}])
            if interrupt_value:
                info = (
                    interrupt_value[0].value
                    if hasattr(interrupt_value[0], "value")
                    else interrupt_value[0]
                )
            else:
                info = {}

            question = (
                info.get("question", "Approve this step?")
                if isinstance(info, dict)
                else str(info)
            )
            step = info.get("step", {}) if isinstance(info, dict) else {}

            print(f"\n{'=' * 60}")
            print(f"APPROVAL REQUIRED: {question}")
            if step:
                print(f"  Agent: {step.get('agent_type', 'unknown')}")
                print(f"  Task: {step.get('task_description', 'N/A')}")
            print(f"{'=' * 60}")

            user_input = input("Approve? [y/n] (with optional feedback): ").strip()
            if user_input.lower().startswith("y"):
                decision = {"decision": "approve", "feedback": None}
            else:
                feedback = user_input[1:].strip() if len(user_input) > 1 else None
                if not feedback:
                    feedback = input("Rejection feedback (optional): ").strip() or None
                decision = {"decision": "reject", "feedback": feedback}

            from langgraph.types import Command

            result_state = self._graph.invoke(Command(resume=decision), config)

        summary = result_state.get("final_summary")
        if summary:
            return summary
        return result_state.get("error", "Orchestration completed with no summary.")

    def run_stream(
        self, user_request: str, *, thread_id: str | None = None
    ) -> Generator[StreamEvent, None, None]:
        """Run the graph, yielding StreamEvents as work progresses."""
        config = self._make_config(thread_id)
        initial_state: OrchestratorState = {
            "user_request": user_request,
            "plan": [],
            "current_step_index": 0,
            "agent_results": [],
            "shared_context": {},
            "pending_approval": False,
            "approval_decision": None,
            "approval_feedback": None,
            "needs_revision": False,
            "revision_feedback": None,
            "final_summary": None,
            "is_complete": False,
            "error": None,
        }

        prev_results_count = 0
        for event in self._graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name == "plan":
                    plan = updates.get("plan", [])
                    yield StreamEvent(
                        kind="text",
                        data=f"Plan created with {len(plan)} steps.",
                    )
                elif node_name == "should_approve":
                    if updates.get("pending_approval"):
                        yield StreamEvent(
                            kind="text",
                            data="Waiting for approval...",
                        )
                elif node_name == "execute_agent":
                    results = updates.get("agent_results", [])
                    for r in results[prev_results_count:]:
                        status = "SUCCESS" if r.get("success") else "FAILED"
                        agent = r.get("agent_type", "unknown")
                        yield StreamEvent(
                            kind="dispatch",
                            data=f"[{status}] {agent}: dispatched",
                        )
                        yield StreamEvent(
                            kind="agent_result",
                            data=r.get("summary", ""),
                        )
                    prev_results_count = len(results)
                elif node_name == "revise":
                    yield StreamEvent(kind="text", data="Revising...")
                elif node_name == "finalize":
                    summary = updates.get("final_summary", "")
                    yield StreamEvent(kind="done", data=summary)

    def resume(
        self,
        thread_id: str,
        *,
        approval_decision: str = "approve",
        approval_feedback: str | None = None,
    ) -> str:
        """Resume an interrupted graph from a checkpoint."""
        from langgraph.types import Command

        config = self._make_config(thread_id)
        decision = {"decision": approval_decision, "feedback": approval_feedback}
        result_state = self._graph.invoke(Command(resume=decision), config)

        summary = result_state.get("final_summary")
        if summary:
            return summary
        return result_state.get("error", "Orchestration completed with no summary.")

    def get_state(self, thread_id: str) -> Any:
        """Introspect the current state for a thread."""
        config = self._make_config(thread_id)
        return self._graph.get_state(config)


# ---------------------------------------------------------------------------
# Module-level graph — required for LangGraph Studio
# ---------------------------------------------------------------------------

try:
    graph = build_graph()
except ImportError:
    graph = None  # LangGraph not installed; Studio won't work but import succeeds
