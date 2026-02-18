"""Tests for the async orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agenticflow.async_orchestrator import AsyncOrchestrator
from agenticflow.models import Workspace


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_response(content, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class _FakeAsyncStream:
    def __init__(self, final_message):
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        async def _empty():
            if False:
                yield None

        return _empty()

    async def get_final_message(self):
        return self._final_message


class TestAsyncOrchestrator:
    def test_run_end_turn(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = AsyncOrchestrator(workspace=ws)
        orch.client = MagicMock()

        response = _make_response([_make_text_block("Async done.")], stop_reason="end_turn")
        with patch(
            "agenticflow.async_orchestrator.async_retry_api_call",
            new=AsyncMock(return_value=response),
        ):
            result = asyncio.run(orch.run("Do async work"))

        assert result == "Async done."

    def test_run_stream_done_contains_summary(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = AsyncOrchestrator(workspace=ws)
        orch.client = MagicMock()

        final_message = _make_response(
            [_make_text_block("Async stream summary.")],
            stop_reason="end_turn",
        )
        orch.client.messages.stream.return_value = _FakeAsyncStream(final_message)

        async def collect():
            events = []
            async for event in orch.run_stream("Stream async work"):
                events.append(event)
            return events

        events = asyncio.run(collect())
        done_events = [e for e in events if e.kind == "done"]
        assert len(done_events) == 1
        assert done_events[0].data == "Async stream summary."

    def test_process_tool_calls_dispatches_in_thread(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = AsyncOrchestrator(workspace=ws)

        tool_use = SimpleNamespace(
            type="tool_use",
            name="dispatch_agent",
            id="tool-1",
            input={"agent_type": "coder", "task_description": "Write code"},
        )

        with patch(
            "agenticflow.async_orchestrator.asyncio.to_thread",
            new=AsyncMock(return_value="[SUCCESS] done"),
        ) as mock_to_thread:
            results = asyncio.run(orch._process_tool_calls([tool_use]))

        mock_to_thread.assert_awaited_once()
        assert results[0]["content"] == "[SUCCESS] done"

    def test_process_tool_calls_invalid_dispatch_sets_error(self, tmp_path: Path):
        ws = Workspace(root=tmp_path / "ws")
        orch = AsyncOrchestrator(workspace=ws)

        tool_use = SimpleNamespace(
            type="tool_use",
            name="dispatch_agent",
            id="tool-1",
            input={"agent_type": "coder"},
        )

        results = asyncio.run(orch._process_tool_calls([tool_use]))
        assert results[0]["is_error"] is True
        assert "dispatch_agent failed" in results[0]["content"]
