"""Researcher node: executes searches and selectively fetches web pages."""
from app.config import get_settings
from app.schemas import SearchResultItem, FetchedPage, ToolCallRecord
from app.state import ResearchState
from app.tools.web_search import WebSearchTool
from app.tools.fetch_page import PageFetchTool
from app.utils.logging_config import setup_logger

logger = setup_logger("node.researcher")


def execute_research_node(state: ResearchState) -> dict:
    settings = get_settings()
    search_tool = WebSearchTool()
    fetch_tool = PageFetchTool()

    plan = state.get("plan")
    queries_to_run = plan.proposed_searches if plan and plan.proposed_searches else [state["question"]]

    all_raw_results: list[SearchResultItem] = list(state.get("raw_search_results", []))
    all_tool_records: list[ToolCallRecord] = list(state.get("tool_records", []))
    all_fetched_pages: list[FetchedPage] = list(state.get("fetched_pages", []))
    errors: list[str] = list(state.get("errors", []))

    existing_urls = {res.url for res in all_raw_results}

    for q in queries_to_run[:2]:
        results, record = search_tool.search(q, max_results=settings.max_search_results_per_query)
        all_tool_records.append(record)
        if record.error_message:
            errors.append(f"Search failed for '{q}': {record.error_message}")

        for item in results:
            if item.url not in existing_urls:
                existing_urls.add(item.url)
                all_raw_results.append(item)

    fetched_urls = {p.url for p in all_fetched_pages}
    candidate_urls = [r.url for r in all_raw_results if r.url not in fetched_urls]
    urls_to_fetch = candidate_urls[:settings.max_pages_to_fetch]

    for url in urls_to_fetch:
        page, record = fetch_tool.fetch(url)
        all_tool_records.append(record)
        all_fetched_pages.append(page)
        if record.error_message:
            errors.append(f"Fetch failed for '{url}': {record.error_message}")

    return {
        "iteration": state.get("iteration", 0) + 1,
        "raw_search_results": all_raw_results,
        "fetched_pages": all_fetched_pages,
        "tool_records": all_tool_records,
        "errors": errors
    }
