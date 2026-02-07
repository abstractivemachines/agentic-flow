"""Tests for retry logic."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from agenticflow.retry import async_retry_api_call, retry_api_call


class TestRetryApiCall:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value="ok")
        result = retry_api_call(fn, max_retries=3)
        assert result == "ok"
        fn.assert_called_once()

    @patch("agenticflow.retry.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep: MagicMock):
        fn = MagicMock()
        fn.side_effect = [
            anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
            "ok",
        ]
        result = retry_api_call(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("agenticflow.retry.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep: MagicMock):
        fn = MagicMock()
        fn.side_effect = [
            anthropic.APIConnectionError(request=MagicMock()),
            "ok",
        ]
        result = retry_api_call(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    @patch("agenticflow.retry.time.sleep")
    def test_retries_on_server_error(self, mock_sleep: MagicMock):
        fn = MagicMock()
        fn.side_effect = [
            anthropic.InternalServerError(
                message="server error",
                response=MagicMock(status_code=500, headers={}),
                body=None,
            ),
            "ok",
        ]
        result = retry_api_call(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"

    @patch("agenticflow.retry.time.sleep")
    def test_exhausts_retries(self, mock_sleep: MagicMock):
        exc = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        fn = MagicMock(side_effect=exc)
        with pytest.raises(anthropic.RateLimitError):
            retry_api_call(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # initial + 2 retries

    def test_non_retryable_error_raised_immediately(self):
        fn = MagicMock(
            side_effect=anthropic.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            )
        )
        with pytest.raises(anthropic.AuthenticationError):
            retry_api_call(fn, max_retries=3)
        fn.assert_called_once()


class TestAsyncRetryApiCall:
    def test_succeeds_first_try(self):
        async def fn():
            return "ok"

        async def run():
            return await async_retry_api_call(fn, max_retries=3)

        result = asyncio.run(run())
        assert result == "ok"

    def test_retries_on_rate_limit(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise anthropic.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                )
            return "ok"

        async def run():
            with patch("agenticflow.retry.asyncio") as mock_asyncio:
                mock_future: asyncio.Future[None] = asyncio.Future()
                mock_future.set_result(None)
                mock_asyncio.sleep.return_value = mock_future
                return await async_retry_api_call(fn, max_retries=3, base_delay=0.01)

        result = asyncio.run(run())
        assert result == "ok"
        assert call_count == 2
