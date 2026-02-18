"""Generic HTTP request tool."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from agenticflow.tools.base import Tool, ToolResult

HTTP_TIMEOUT = 15


def _is_local_or_private_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_outbound_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("URL must include a valid hostname.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed.")

    host = parsed.hostname.lower()
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
    ):
        raise ValueError(f"Blocked local hostname: {host}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None and _is_local_or_private_ip(ip):
        raise ValueError(f"Blocked local/private IP address: {ip}")

    if ip is None:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            # Let request path raise a useful resolution error.
            return

        for info in infos:
            resolved_ip_str = info[4][0].split("%", 1)[0]
            try:
                resolved_ip = ipaddress.ip_address(resolved_ip_str)
            except ValueError:
                continue
            if _is_local_or_private_ip(resolved_ip):
                raise ValueError(
                    f"Blocked URL because hostname resolves to local/private IP: {resolved_ip}"
                )


class HttpRequestTool(Tool):
    """Make HTTP requests to URLs."""

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return "Make an HTTP request to a URL and return the response."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, PATCH).",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                },
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to.",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH).",
                },
                "headers": {
                    "type": "object",
                    "description": "Additional request headers.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["method", "url"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        method: str = kwargs["method"]
        url: str = kwargs["url"]
        body: str | None = kwargs.get("body")
        headers: dict[str, str] = kwargs.get("headers", {})

        try:
            _validate_outbound_url(url)
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                resp = client.request(
                    method=method,
                    url=url,
                    content=body,
                    headers=headers,
                )
            # Truncate very large responses
            body_text = resp.text[:10_000]
            if len(resp.text) > 10_000:
                body_text += "\n... (truncated)"
            return ToolResult(
                output=f"HTTP {resp.status_code}\n\n{body_text}",
                is_error=resp.status_code >= 400,
            )
        except Exception as e:
            return ToolResult(output=f"HTTP request failed: {e}", is_error=True)
