# AgenticFlow

Multi-agent software engineering system powered by Claude.

AgenticFlow uses an orchestrator pattern to break down complex software engineering tasks and dispatch them to specialized agents — planner, coder, reviewer, and tester — each with their own tools and system prompts.

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│ Orchestrator │ ── Claude decides which agent to invoke
└─────┬───────┘
      │  dispatch_agent tool calls
      │
      ├─── Backend: api ──────── Direct Anthropic API calls (ANTHROPIC_API_KEY)
      ├─── Backend: claude-code ─ Claude Code CLI via claude-agent-sdk (no key needed)
      ├─── Backend: langgraph ── LangGraph with checkpointing, HITL, Studio debugging
      │
      ▼
┌──────────┬──────────┬──────────┐
│ Planner  │  Coder   │ Reviewer │  Tester
│          │          │          │
│ research │ read/    │ read/    │ read/write/
│ plan     │ write/   │ run cmds │ run tests
│          │ run cmds │          │
└──────────┴──────────┴──────────┘
      │
      ▼
  Workspace (sandboxed directory)
```

**Agents:**

| Agent    | Role                           | Tools                              |
|----------|--------------------------------|------------------------------------|
| Planner  | Architecture & design          | read_file, list_dir, web_search, http |
| Coder    | Write & modify code            | read_file, write_file, list_dir, run_command, git, shell |
| Reviewer | Code review & quality checks   | read_file, list_dir, run_command   |
| Tester   | Write & run tests              | read_file, write_file, list_dir, run_command |

## Setup

```bash
# Clone
git clone https://github.com/abstractivemachines/agentic-flow.git
cd agentic-flow

# Install (pick the backends you need)
pip install -e ".[dev,claude-code,langgraph]"

# Set your API key (only needed for the api backend)
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### CLI

```bash
# Simple task (uses api backend by default)
agenticflow "Create a Python function that checks if a number is prime"

# Specify model and workspace
agenticflow "Build a REST API with Flask" --model claude-sonnet-4-5-20250929 --workspace ./my-project

# Use Claude Code backend (no API key needed — authenticates via Claude Code CLI)
agenticflow "Build a REST API with Flask" --backend claude-code

# Use LangGraph backend with checkpointing and no approval gates
agenticflow "Build a REST API with Flask" --backend langgraph --no-approve

# LangGraph with human-in-the-loop approval (gates all agents)
agenticflow "Build a REST API with Flask" --backend langgraph --approval-policy strict

# Resume an interrupted LangGraph run
agenticflow "ignored" --backend langgraph --thread-id abc123

# Verbose mode
agenticflow "Refactor the auth module" -v
```

| Flag | Description | Default |
|------|-------------|---------|
| `task` | Task description (positional) | — |
| `--backend` | `api`, `claude-code`, or `langgraph` | `api` |
| `--model` | Claude model ID | `claude-sonnet-4-5-20250929` |
| `--workspace` | Working directory for agents | `./workspace` |
| `-v, --verbose` | Enable verbose logging | off |
| `--checkpoint-backend` | Checkpoint store for langgraph (`sqlite`, `memory`) | `sqlite` |
| `--approval-policy` | HITL approval policy for langgraph (`none`, `default`, `strict`) | `none` |
| `--no-approve` | Shorthand for `--approval-policy none` | — |
| `--thread-id` | Thread ID to resume a langgraph run | — |

### Python API

```python
from pathlib import Path
from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator

workspace = Workspace(root=Path("./my-project"))
orchestrator = Orchestrator(workspace=workspace, verbose=True)
result = orchestrator.run("Build a CLI tool that converts CSV to JSON")
print(result)
```

### Async API

```python
import asyncio
from pathlib import Path
from agenticflow.models import Workspace
from agenticflow.async_orchestrator import AsyncOrchestrator

async def main():
    workspace = Workspace(root=Path("./my-project"))
    orchestrator = AsyncOrchestrator(workspace=workspace)
    result = await orchestrator.run("Build a REST API")
    print(result)

asyncio.run(main())
```

### Streaming

```python
from pathlib import Path
from agenticflow.models import Workspace
from agenticflow.orchestrator import Orchestrator

workspace = Workspace(root=Path("./my-project"))
orchestrator = Orchestrator(workspace=workspace)

for event in orchestrator.run_stream("Build a web scraper"):
    print(event)
```

### Per-Agent Model Configuration

```python
orchestrator = Orchestrator(
    workspace=workspace,
    model="claude-sonnet-4-5-20250929",        # default model
    agent_models={
        "planner": "claude-opus-4-6",           # use Opus for planning
        "coder": "claude-sonnet-4-5-20250929",  # Sonnet for coding
        "reviewer": "claude-opus-4-6",          # Opus for reviews
        "tester": "claude-sonnet-4-5-20250929", # Sonnet for tests
    },
)
```

### Claude Code Backend

The `claude-code` backend delegates orchestration and tool execution to the Claude Code CLI via `claude-agent-sdk`. No `ANTHROPIC_API_KEY` is needed — authentication is handled by the CLI.

```python
import asyncio
from pathlib import Path
from agenticflow.claude_code import ClaudeCodeOrchestrator
from agenticflow.models import Workspace

async def main():
    workspace = Workspace(root=Path("./my-project"))
    orchestrator = ClaudeCodeOrchestrator(
        workspace=workspace,
        model="claude-sonnet-4-5-20250929",
        agent_models={"planner": "claude-opus-4-6"},
        verbose=True,
    )
    result = await orchestrator.run("Build a REST API")
    print(result)

asyncio.run(main())
```

Streaming is also supported:

```python
async for event in orchestrator.run_stream("Build a web scraper"):
    if event.kind == "text":
        print(event.data, end="", flush=True)
```

### LangGraph Backend

The `langgraph` backend wraps the same agents in a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`, adding checkpointing, human-in-the-loop approval gates, and LangGraph Studio visual debugging. Requires `ANTHROPIC_API_KEY`.

```python
from pathlib import Path
from agenticflow.langgraph_orchestrator import LangGraphOrchestrator
from agenticflow.models import Workspace

workspace = Workspace(root=Path("./my-project"))
orchestrator = LangGraphOrchestrator(
    workspace=workspace,
    checkpoint_backend="sqlite",   # or "memory" for tests
    approval_policy="default",     # gates coder agents; "strict" gates all, "none" gates nothing
    verbose=True,
)
result = orchestrator.run("Build a REST API", thread_id="my-thread")
print(result)
```

**Checkpointing** — State is persisted between nodes (SQLite by default at `{workspace}/.agenticflow/checkpoints.db`). If the process dies, resume from the last checkpoint:

```python
result = orchestrator.resume("my-thread")
```

**Human-in-the-loop** — When the approval policy gates an agent, the graph interrupts and waits. In the CLI this is an interactive prompt; programmatically use `resume()`:

```python
result = orchestrator.resume(
    "my-thread",
    approval_decision="approve",    # or "reject"
    approval_feedback="Looks good",
)
```

**Streaming:**

```python
for event in orchestrator.run_stream("Build a web scraper"):
    if event.kind == "done":
        print(event.data)
```

**LangGraph Studio** — A `langgraph.json` is included at the project root. Point LangGraph Studio at this directory to visually debug the orchestrator graph.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## Docker

```bash
docker build -t agenticflow .
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY agenticflow "Build a hello world app"
```

## License

MIT
