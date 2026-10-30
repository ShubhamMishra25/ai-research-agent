"""Researcher node: selects tools, executes searches, and fetches web pages."""
from app.config import get_settings
from app.schemas import SearchResultItem, FetchedPage, ToolCallRecord
from app.state import ResearchState
from app.tools.web_search import WebSearchTool
from app.tools.fetch_page import PageFetchTool
from app.nodes.planner import generate_follow_up_queries
from app.utils.logging_config import setup_logger
from app.utils.metrics import track_node
from app.utils.urls import canonicalize_url, extract_direct_urls

logger = setup_logger("node.researcher")

KNOWN_TOOLS = {"web_search", "fetch_page"}
OFFICIAL_DOMAIN_HINTS = {
    "openai": ("openai.com", "platform.openai.com", "help.openai.com"),
    "google": ("google.com", "deepmind.google", "ai.google"),
    "microsoft": ("microsoft.com", "blogs.microsoft.com"),
    "anthropic": ("anthropic.com",),
    "meta": ("meta.com", "ai.meta.com"),
}


def _selected_tools(state: ResearchState) -> set[str]:
    plan = state.get("plan")
    intended = {tool.strip().lower() for tool in (plan.intended_tools if plan else []) if isinstance(tool, str)}
    selected = intended & KNOWN_TOOLS
    if not selected:
        selected = set(KNOWN_TOOLS)
    return selected


def _direct_url_results(question: str, plan) -> list[SearchResultItem]:
    urls = extract_direct_urls(question)
    if plan and plan.objective:
        urls.extend(extract_direct_urls(plan.objective))
    unique = []
    seen = set()
    for url in urls:
        key = canonicalize_url(url)
        if key not in seen:
            seen.add(key)
            unique.append(SearchResultItem(title="Direct URL from question", url=url, snippet="", score=1.0))
    return unique


def _authority_hint(url: str, question: str) -> int:
    lowered_question = question.lower()
    lowered_url = url.lower()
    return int(any(
        organization in lowered_question and domain in lowered_url
        for organization, domains in OFFICIAL_DOMAIN_HINTS.items()
        for domain in domains
    ))


@track_node("researcher")
def execute_research_node(state: ResearchState) -> dict:
    settings = get_settings()
    search_tool = WebSearchTool()
    fetch_tool = PageFetchTool()
    execution_id = state.get("execution_id")

    plan = state.get("plan")
    selected_tools = _selected_tools(state)

    all_raw_results: list[SearchResultItem] = list(state.get("raw_search_results", []))
    all_tool_records: list[ToolCallRecord] = list(state.get("tool_records", []))
    all_fetched_pages: list[FetchedPage] = list(state.get("fetched_pages", []))
    errors: list[str] = list(state.get("errors", []))
    used_queries = list(state.get("used_queries", []))
    used_query_keys = {query.casefold().strip() for query in used_queries}

    existing_keys = {canonicalize_url(res.url) for res in all_raw_results}
    for direct in _direct_url_results(state.get("question", ""), plan):
        key = canonicalize_url(direct.url)
        if key not in existing_keys:
            existing_keys.add(key)
            all_raw_results.append(direct)

    if state.get("iteration", 0) > 0:
        queries_to_run = generate_follow_up_queries(state)
    else:
        queries_to_run = list(plan.proposed_searches) if plan and plan.proposed_searches else [state["question"]]
        # Planned subquestions make useful searches too, especially for
        # entity-discovery queries where one generic query misses candidates.
        if plan:
            existing = {query.casefold().strip() for query in queries_to_run}
            for subquestion in plan.subquestions:
                if len(queries_to_run) >= 3:
                    break
                key = subquestion.casefold().strip()
                if key and key not in existing:
                    existing.add(key)
                    queries_to_run.append(subquestion)

    if not queries_to_run:
        # Never burn an iteration without doing work.
        planned = [query for query in (plan.proposed_searches if plan else []) if query.strip()]
        queries_to_run = [
            query for query in planned if query.casefold().strip() not in used_query_keys
        ] or [f"{state['question']} detailed authoritative sources"]

    searches_run = 0
    # Follow-up iterations escalate to search because they are gap-driven.
    use_search = "web_search" in selected_tools or state.get("iteration", 0) > 0
    if use_search:
        for q in queries_to_run[:3]:
            normalized_query = q.strip()
            if not normalized_query or normalized_query.casefold() in used_query_keys:
                continue
            used_query_keys.add(normalized_query.casefold())
            used_queries.append(normalized_query)
            results, record = search_tool.search(
                normalized_query, max_results=settings.max_search_results_per_query
            )
            searches_run += 1
            all_tool_records.append(record)
            if record.error_message:
                errors.append(f"Search failed for '{normalized_query}': {record.error_message}")

            for item in results:
                key = canonicalize_url(item.url)
                if key not in existing_keys:
                    existing_keys.add(key)
                    all_raw_results.append(item)

    pages_fetched = 0
    if "fetch_page" in selected_tools:
        # Retry previously failed pages, but skip anything successfully fetched.
        fetched_keys = {
            canonicalize_url(page.url)
            for page in all_fetched_pages
            if page.error is None and page.char_count > 0
        }
        score_by_url = {result.url: result.score or 0.0 for result in all_raw_results}
        candidate_urls = [
            result.url
            for result in all_raw_results
            if canonicalize_url(result.url) not in fetched_keys
        ]
        candidate_urls.sort(
            key=lambda url: (
                _authority_hint(url, state.get("question", "")),
                score_by_url.get(url, 0.0),
            ),
            reverse=True,
        )
        urls_to_fetch = candidate_urls[: settings.max_pages_to_fetch]

        for url in urls_to_fetch:
            page, record = fetch_tool.fetch(url)
            pages_fetched += 1
            all_tool_records.append(record)
            url_key = canonicalize_url(url)
            all_fetched_pages = [
                existing for existing in all_fetched_pages
                if canonicalize_url(existing.url) != url_key
            ]
            all_fetched_pages.append(page)
            if record.error_message:
                errors.append(f"Fetch failed for '{url}': {record.error_message}")

    logger.info(
        "Research iteration %d executed (tools=%s, searches=%d, pages=%d)",
        state.get("iteration", 0) + 1,
        sorted(selected_tools),
        searches_run,
        pages_fetched,
        extra={"execution_id": execution_id},
    )

    return {
        "iteration": state.get("iteration", 0) + 1,
        "used_queries": used_queries,
        "raw_search_results": all_raw_results,
        "fetched_pages": all_fetched_pages,
        "tool_records": all_tool_records,
        "errors": errors,
        "_metrics": {
            "tools_selected": sorted(selected_tools),
            "searches_run": searches_run,
            "pages_fetched": pages_fetched,
        },
    }
