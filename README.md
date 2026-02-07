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

# Install
pip install -e ".[dev]"

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### CLI

```bash
# Simple task
agenticflow "Create a Python function that checks if a number is prime"

# Specify model and workspace
agenticflow "Build a REST API with Flask" --model claude-sonnet-4-5-20250929 --workspace ./my-project

# Verbose mode
agenticflow "Refactor the auth module" -v
```

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
