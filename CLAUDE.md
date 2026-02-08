# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
pip install -e ".[dev]"           # Install with dev deps
pytest                             # Run all tests
pytest tests/test_tools.py         # Run a specific test file
pytest tests/test_tools.py -k test_read_file  # Run a single test
ruff check src/ tests/             # Lint
ruff format src/ tests/            # Format
ruff format --check src/ tests/    # Check formatting without modifying
```

Requires `ANTHROPIC_API_KEY` env var at runtime (not for tests). Python 3.11+.

## Architecture

AgenticFlow is a multi-agent orchestration system where Claude acts as both the orchestrator and the individual agents.

**Request flow:** User request → Orchestrator (Claude with `dispatch_agent` + `set_shared_context` tools) → Specialist agents (each running their own agentic tool-use loop) → Results collected in workspace.

**Orchestrator** (`orchestrator.py` / `async_orchestrator.py`): Top-level dispatch loop. Calls Claude with two meta-tools: `dispatch_agent` (invokes a specialist) and `set_shared_context` (stores data for inter-agent communication). Max 20 orchestrator turns. Supports per-agent model overrides via `agent_models` dict. `run_stream()` yields `StreamEvent` objects (kinds: text, dispatch, agent_result, done, error).

**Agents** (`agents/`): Each agent extends `Agent` ABC from `agents/base.py`. The base class implements the agentic tool-use loop (max 25 turns). Each specialist has a role-specific system prompt and tool set:
- **Planner** — architecture/design; tools: read_file, list_directory, web_search, http_request
- **Coder** — code writing; tools: read_file, write_file, list_directory, run_command, git, shell (max_tokens: 8192)
- **Reviewer** — code review; tools: read_file, list_directory, run_command; outputs APPROVE/REQUEST_CHANGES
- **Tester** — test writing/running; tools: read_file, write_file, list_directory, run_command

**Tools** (`tools/`): Each tool implements the `Tool` ABC from `tools/base.py`. The registry in `tools/__init__.py` (`get_tools_for_agent()`) maps agent types to tool instances. All file tools sandbox paths via `_resolve()` to prevent traversal attacks.

**Models** (`models.py`): `Task`, `TaskResult`, `TaskStatus`, `AgentType`, `Workspace`, `SharedContext`. `Workspace.resolve()` enforces path sandboxing. `SharedContext` is thread-safe (uses `threading.Lock`).

**Retry** (`retry.py`): `retry_api_call` / `async_retry_api_call` with exponential backoff + jitter for transient API errors (connection, rate limit, 5xx). Non-retryable errors (auth, bad request) propagate immediately.

## Code Conventions

- `from __future__ import annotations` in every module
- Type hints everywhere; dataclasses for models, ABCs for interfaces
- All file I/O sandboxed to workspace via `Workspace.resolve()`
- Tools implement `Tool` ABC (`tools/base.py`); agents extend `Agent` ABC (`agents/base.py`)
- Tests use `unittest.mock` — no real API calls. Mock helpers create `SimpleNamespace` objects mimicking Anthropic response shapes (`_make_text_block`, `_make_tool_use_block`, `_make_response`)

## Test Patterns

Tests mock `agent.client` or `orchestrator.client` with `MagicMock`. Multi-turn agent interactions use `side_effect` lists on `client.messages.create`. Use `tmp_path` fixture for temporary workspaces. See `tests/test_orchestrator.py` for the mock response helper pattern.

## Post-Change Checklist

After implementing any feature, fix, or refactor, always consider updating:
- **README.md** — CLI flags, Python API examples, architecture diagram, setup instructions
- **CLAUDE.md** — Architecture descriptions, code conventions, test patterns
