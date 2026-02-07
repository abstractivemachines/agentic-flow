# CLAUDE.md

## Project Overview

AgenticFlow is a multi-agent software engineering system powered by Claude. It uses an orchestrator that dispatches tasks to specialized agents (planner, coder, reviewer, tester).

## Build & Test

```bash
pip install -e ".[dev]"    # Install with dev deps
pytest                      # Run all tests
pytest tests/test_tools.py  # Run specific test file
ruff check src/ tests/      # Lint
ruff format src/ tests/     # Format
```

## Code Conventions

- Python 3.11+ with type hints everywhere
- Dataclasses for models, ABCs for interfaces
- `from __future__ import annotations` in all modules
- All file I/O sandboxed to workspace via `Workspace.resolve()`
- Tools implement the `Tool` ABC from `tools/base.py`
- Agents extend `Agent` ABC from `agents/base.py`
- Tests use `unittest.mock` to mock the Anthropic API — no real API calls in tests
- Use `ruff` for linting and formatting (configured in pyproject.toml)

## Architecture

- `orchestrator.py` / `async_orchestrator.py` — top-level dispatch loop
- `agents/` — specialist agents with different system prompts and tool sets
- `tools/` — sandboxed tool implementations; `__init__.py` has the registry
- `models.py` — Task, TaskResult, Workspace, SharedContext dataclasses
