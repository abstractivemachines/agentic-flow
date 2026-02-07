"""Generic HTTP request tool."""

from __future__ import annotations

from typing import Any

import httpx

from agenticflow.tools.base import Tool, ToolResult

HTTP_TIMEOUT = 15


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
