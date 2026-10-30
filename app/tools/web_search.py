"""Production web search tool with retries, error handling, and normalization."""
import time
from typing import List, Optional, Tuple

from app.config import get_settings
from app.schemas import SearchResultItem, ToolCallRecord
from app.utils.logging_config import setup_logger
from app.utils.retry import with_retries

logger = setup_logger("tools.web_search")


class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = self.settings.tavily_api_key if api_key is None else api_key

    def search(self, query: str, max_results: int = 4) -> Tuple[List[SearchResultItem], ToolCallRecord]:
        start = time.perf_counter()
        if not self.api_key:
            record = ToolCallRecord(
                tool_name="web_search",
                query_or_url=query,
                status="FAILURE",
                latency_ms=(time.perf_counter() - start) * 1000,
                error_message="Tavily API key is not configured",
            )
            return [], record

        attempts_made = 0

        def _search_once():
            nonlocal attempts_made
            attempts_made += 1
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)
            return client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_raw_content=False,
                timeout=self.settings.request_timeout_seconds,
            )

        try:
            response = with_retries(
                _search_once,
                attempts=self.settings.max_retries + 1,
                backoff_seconds=self.settings.retry_backoff_seconds,
                description=f"web_search '{query}'",
            )

            results: List[SearchResultItem] = []
            seen_urls = set()

            raw_results = response.get("results", []) if isinstance(response, dict) else []
            for item in raw_results if isinstance(raw_results, list) else []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    score = float(item.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                title = item.get("title") or "Untitled Source"
                content = item.get("content") or ""
                results.append(
                    SearchResultItem(
                        title=str(title).strip() or "Untitled Source",
                        url=url,
                        snippet=str(content).strip(),
                        score=score,
                        published_date=(
                            str(item["published_date"]).strip()
                            if item.get("published_date")
                            else None
                        ),
                    )
                )

            latency = (time.perf_counter() - start) * 1000
            record = ToolCallRecord(
                tool_name="web_search",
                query_or_url=query,
                status="SUCCESS",
                latency_ms=round(latency, 2),
                attempts=attempts_made,
            )
            return results, record

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            err_msg = f"Web search failed: {type(exc).__name__} - {str(exc)}"
            logger.error(err_msg, exc_info=True)
            record = ToolCallRecord(
                tool_name="web_search",
                query_or_url=query,
                status="FAILURE",
                latency_ms=round(latency, 2),
                attempts=max(1, attempts_made),
                error_message=err_msg,
            )
            return [], record
