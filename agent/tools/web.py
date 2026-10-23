"""Web tools: fetch a URL and run a web search."""
from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests
from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.security import SecurityError, SecurityPolicy

SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}
BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "ul", "ol",
}


class _TextExtractor(HTMLParser):
    """Minimal HTML -> plain text converter (no bs4 dependency)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._href: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_text = []
        if tag in BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "a" and self._href:
            label = " ".join("".join(self._link_text).split())
            if label:
                self.links.append((label, self._href))
            self._href = None
            self._link_text = []
        if tag in BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
            return
        if self._skip_depth:
            return
        if self._href is not None:
            self._link_text.append(data)
        text = data.strip()
        if text:
            self.chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self.chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _extract_links(parser: _TextExtractor, limit: int) -> str:
    if not parser.links:
        return ""
    rows = [f"- {label} -> {href}" for label, href in parser.links[:limit]]
    return "\n".join(rows)


def web_tools(settings: Settings, policy: SecurityPolicy) -> list[BaseTool]:
    """Build web-access tools."""

    @tool
    def fetch_url(url: str, max_chars: int = 12000) -> str:
        """Download a web page and return its readable text content.

        Loopback, private-network and cloud-metadata addresses are blocked.

        Args:
            url: Absolute http(s) URL to fetch.
            max_chars: Maximum number of characters of page text to return.
        """
        try:
            policy.check_url(url)
        except SecurityError as exc:
            policy.audit("fetch_url_blocked", url=url)
            return f"ERROR: {exc}"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; custom-agent/0.1)",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        }
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=settings.request_timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return f"ERROR: failed to fetch {url}: {exc}"

        # Re-check after redirects: a public URL can bounce to an internal one.
        try:
            policy.check_url(response.url)
        except SecurityError as exc:
            policy.audit("fetch_url_blocked", url=url, final=response.url)
            return f"ERROR: redirect target blocked: {exc}"

        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            body = response.text[:max_chars]
            policy.audit("fetch_url", url=url)
            return policy.redact(f"URL: {url}\nContent-Type: {content_type}\n\n{body}")

        parser = _TextExtractor()
        try:
            parser.feed(response.text)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: could not parse HTML from {url}: {exc}"

        text = parser.text()
        text = re.sub(r"\n{3,}", "\n\n", text)
        header = f"URL: {url}\nTitle: {parser.title or '(none)'}\n"
        links = _extract_links(parser, 40)
        if links:
            header += f"\nLinks:\n{links}\n"
        body = text[:max_chars]
        if len(text) > max_chars:
            body += f"\n... [truncated {len(text) - max_chars} chars]"
        policy.audit("fetch_url", url=url)
        return policy.redact(header + "\n" + body)

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web and return result titles, URLs and snippets.

        Requires a search API key in the environment: set TAVILY_API_KEY or
        SERPER_API_KEY in .env.

        Args:
            query: The search query.
            max_results: How many results to return.
        """
        if not policy.allow_network:
            return "ERROR: network access is disabled for this session (--offline)."

        max_results = min(max(int(max_results), 1), 20)
        tavily_key = os.getenv("TAVILY_API_KEY")
        serper_key = os.getenv("SERPER_API_KEY")

        if tavily_key:
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                    timeout=settings.request_timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                return f"ERROR: Tavily search failed: {exc}"
            data = response.json()
            results = data.get("results", [])
            if not results:
                return f"No results for '{query}'."
            lines = [f"Search: {query}"]
            for item in results[:max_results]:
                lines.append(
                    f"- {item.get('title', '(untitled)')}\n  {item.get('url', '')}\n"
                    f"  {html.unescape(item.get('content', '') or '')[:400]}"
                )
            policy.audit("web_search", query=query, provider="tavily")
            return policy.redact("\n".join(lines))

        if serper_key:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                    timeout=settings.request_timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                return f"ERROR: Serper search failed: {exc}"
            data = response.json()
            lines = [f"Search: {query}"]
            for item in data.get("organic", [])[:max_results]:
                lines.append(
                    f"- {item.get('title', '(untitled)')}\n  {item.get('link', '')}\n"
                    f"  {html.unescape(item.get('snippet', '') or '')[:400]}"
                )
            policy.audit("web_search", query=query, provider="serper")
            return policy.redact("\n".join(lines) if len(lines) > 1 else f"No results for '{query}'.")

        return (
            "ERROR: web search is not configured. Add TAVILY_API_KEY or SERPER_API_KEY "
            "to your .env file to enable this tool."
        )

    return [fetch_url, web_search]
