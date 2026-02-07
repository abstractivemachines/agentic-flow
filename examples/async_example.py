"""Example: using the async orchestrator."""

import asyncio
from pathlib import Path

from agenticflow.async_orchestrator import AsyncOrchestrator
from agenticflow.models import Workspace


async def main() -> None:
    workspace = Workspace(root=Path("./example_workspace"))
    orchestrator = AsyncOrchestrator(
        workspace=workspace,
        model="claude-sonnet-4-5-20250929",
        verbose=True,
    )

    result = await orchestrator.run(
        "Create a Python script that downloads a webpage and counts word frequencies."
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
