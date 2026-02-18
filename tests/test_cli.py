"""Tests for CLI argument behavior."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agenticflow.cli import main


class TestCli:
    def test_langgraph_resume_without_task(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "agenticflow.langgraph_orchestrator.LangGraphOrchestrator"
            ) as mock_orch_cls:
                orch = MagicMock()
                orch.resume.return_value = "resumed"
                mock_orch_cls.return_value = orch

                with patch("builtins.print"):
                    main(["--backend", "langgraph", "--thread-id", "abc123"])

                orch.resume.assert_called_once_with("abc123")

    def test_task_required_without_resume(self):
        with pytest.raises(SystemExit):
            main(["--backend", "langgraph"])

    def test_langgraph_run_with_task_and_thread_id_uses_run(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "agenticflow.langgraph_orchestrator.LangGraphOrchestrator"
            ) as mock_orch_cls:
                orch = MagicMock()
                orch.run.return_value = "ran"
                mock_orch_cls.return_value = orch

                with patch("builtins.print"):
                    main(
                        [
                            "--backend",
                            "langgraph",
                            "--thread-id",
                            "thread-xyz",
                            "build api",
                        ]
                    )

                orch.run.assert_called_once_with("build api", thread_id="thread-xyz")
                orch.resume.assert_not_called()
