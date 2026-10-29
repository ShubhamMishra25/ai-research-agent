"""Evidence processing node: filters, deduplicates, and assesses sufficiency."""
from app.schemas import FilteredEvidence
from app.state import ResearchState
from app.utils.logging_config import setup_logger

logger = setup_logger("node.evidence")


def process_evidence_node(state: ResearchState) -> dict:
    raw_results = state.get("raw_search_results", [])
    fetched_pages = {p.url: p for p in state.get("fetched_pages", []) if p.char_count > 50}

    filtered: list[FilteredEvidence] = []
    source_index = 1

    for res in raw_results:
        page = fetched_pages.get(res.url)
        if page:
            text_snippet = page.extracted_text[:1200]
            is_full = True
        else:
            text_snippet = res.snippet
            is_full = False

        if not text_snippet or len(text_snippet.strip()) < 15:
            continue

        evidence_item = FilteredEvidence(
            source_id=f"S{source_index}",
            url=res.url,
            title=page.title if page and page.title != "Untitled" else res.title,
            summary_or_excerpt=text_snippet,
            is_full_page=is_full,
            relevance_score=1.0 if is_full else 0.7
        )
        filtered.append(evidence_item)
        source_index += 1

    current_iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    is_sufficient = len(filtered) >= 2 or (len(filtered) == 1 and any(e.is_full_page for e in filtered))

    if not is_sufficient:
        if current_iteration < max_iterations:
            reason = f"Insufficient evidence gathered ({len(filtered)} valid sources). Triggering refinement."
        else:
            reason = f"Budget exhausted after {current_iteration} iterations with limited evidence."
    else:
        reason = f"Sufficient evidence gathered ({len(filtered)} validated sources)."

    logger.info(f"Evidence status: {reason}")

    return {
        "evidence": filtered,
        "is_sufficient": is_sufficient,
        "sufficiency_reason": reason
    }
