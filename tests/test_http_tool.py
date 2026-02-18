"""Tests for the HTTP request tool."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from agenticflow.tools.http_client import HttpRequestTool


def _dns_result(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 0))]


class TestHttpRequestTool:
    @patch("agenticflow.tools.http_client.socket.getaddrinfo")
    @patch("agenticflow.tools.http_client.httpx.Client")
    def test_allows_public_https(self, mock_client_cls: MagicMock, mock_getaddrinfo: MagicMock):
        mock_getaddrinfo.return_value = _dns_result("93.184.216.34")

        response = MagicMock()
        response.status_code = 200
        response.text = "ok"
        mock_client = MagicMock()
        mock_client.request.return_value = response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = HttpRequestTool()
        result = tool.execute(method="GET", url="https://example.com")

        assert result.is_error is False
        assert "HTTP 200" in result.output
        mock_client.request.assert_called_once()

    @patch("agenticflow.tools.http_client.httpx.Client")
    def test_blocks_localhost_hostname(self, mock_client_cls: MagicMock):
        tool = HttpRequestTool()
        result = tool.execute(method="GET", url="http://localhost:8080")

        assert result.is_error is True
        assert "Blocked local hostname" in result.output
        mock_client_cls.assert_not_called()

    @patch("agenticflow.tools.http_client.httpx.Client")
    def test_blocks_private_ip_literal(self, mock_client_cls: MagicMock):
        tool = HttpRequestTool()
        result = tool.execute(method="GET", url="http://127.0.0.1:8000")

        assert result.is_error is True
        assert "Blocked local/private IP" in result.output
        mock_client_cls.assert_not_called()

    @patch("agenticflow.tools.http_client.socket.getaddrinfo")
    @patch("agenticflow.tools.http_client.httpx.Client")
    def test_blocks_hostname_resolving_to_private_ip(
        self, mock_client_cls: MagicMock, mock_getaddrinfo: MagicMock
    ):
        mock_getaddrinfo.return_value = _dns_result("10.0.0.5")

        tool = HttpRequestTool()
        result = tool.execute(method="GET", url="http://public.example")

        assert result.is_error is True
        assert "resolves to local/private IP" in result.output
        mock_client_cls.assert_not_called()

    @patch("agenticflow.tools.http_client.httpx.Client")
    def test_blocks_non_http_scheme(self, mock_client_cls: MagicMock):
        tool = HttpRequestTool()
        result = tool.execute(method="GET", url="file:///etc/passwd")

        assert result.is_error is True
        assert "Only http:// and https:// URLs are allowed" in result.output
        mock_client_cls.assert_not_called()
