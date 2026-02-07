"""Web search tool using httpx."""

from __future__ import annotations

from typing import Any

import httpx

from agenticflow.tools.base import Tool, ToolResult

SEARCH_TIMEOUT = 15


class WebSearchTool(Tool):
    """Search the web and return results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns search result snippets."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs["query"]
        try:
            # Use DuckDuckGo HTML search (no API key required)
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "AgenticFlow/0.1"},
                timeout=SEARCH_TIMEOUT,
                follow_redirects=True,
            )
            resp.raise_for_status()
            # Extract text snippets from the response (simple parsing)
            from html.parser import HTMLParser

            class SnippetParser(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self.in_result = False
                    self.snippets: list[str] = []
                    self._current: list[str] = []

                def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                    attr_dict = dict(attrs)
                    if attr_dict.get("class") and "result__snippet" in attr_dict["class"]:
                        self.in_result = True
                        self._current = []

                def handle_endtag(self, tag: str) -> None:
                    if self.in_result and tag in ("a", "span", "td"):
                        text = "".join(self._current).strip()
                        if text:
                            self.snippets.append(text)
                        self.in_result = False

                def handle_data(self, data: str) -> None:
                    if self.in_result:
                        self._current.append(data)

            parser = SnippetParser()
            parser.feed(resp.text)

            if parser.snippets:
                results = "\n\n".join(
                    f"{i+1}. {s}" for i, s in enumerate(parser.snippets[:5])
                )
                return ToolResult(output=f"Search results for '{query}':\n\n{results}")
            return ToolResult(output=f"No results found for '{query}'.")
        except Exception as e:
            return ToolResult(output=f"Search failed: {e}", is_error=True)
