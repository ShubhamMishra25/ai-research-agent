"""Production web search tool with error handling, normalization, and deduplication."""
import time
from typing import List, Optional, Tuple
from app.config import get_settings
from app.schemas import SearchResultItem, ToolCallRecord
from app.utils.logging_config import setup_logger

logger = setup_logger("tools.web_search")


class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = get_settings().tavily_api_key if api_key is None else api_key

    def search(self, query: str, max_results: int = 4) -> Tuple[List[SearchResultItem], ToolCallRecord]:
        start = time.perf_counter()
        if not self.api_key:
            record = ToolCallRecord(
                tool_name="web_search",
                query_or_url=query,
                status="FAILURE",
                latency_ms=(time.perf_counter() - start) * 1000,
                error_message="Tavily API key is not configured"
            )
            return [], record

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_raw_content=False
            )

            results: List[SearchResultItem] = []
            seen_urls = set()

            for item in response.get("results", []):
                url = item.get("url", "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(SearchResultItem(
                    title=item.get("title", "Untitled Source").strip(),
                    url=url,
                    snippet=item.get("content", "").strip(),
                    score=float(item.get("score", 0.0))
                ))

            latency = (time.perf_counter() - start) * 1000
            record = ToolCallRecord(
                tool_name="web_search",
                query_or_url=query,
                status="SUCCESS",
                latency_ms=round(latency, 2)
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
                error_message=err_msg
            )
            return [], record
