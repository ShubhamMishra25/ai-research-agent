"""Evidence processing node: filters, deduplicates, and assesses sufficiency."""
import re
from urllib.parse import urlparse

from app.schemas import FilteredEvidence
from app.state import ResearchState
from app.utils.logging_config import setup_logger

logger = setup_logger("node.evidence")

STOPWORDS = {
    "about", "after", "also", "and", "are", "for", "from", "how", "into",
    "that", "the", "their", "these", "this", "what", "which", "with",
}


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in STOPWORDS
    }


def process_evidence_node(state: ResearchState) -> dict:
    raw_results = state.get("raw_search_results", [])
    fetched_pages = {p.url: p for p in state.get("fetched_pages", []) if p.char_count > 50}

    filtered: list[FilteredEvidence] = []
    question_terms = _terms(state.get("question", ""))
    plan = state.get("plan")
    planned_terms = _terms(" ".join(plan.subquestions)) if plan else set()
    target_terms = question_terms | planned_terms
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

        evidence_terms = _terms(f"{res.title} {res.snippet} {text_snippet}")
        overlap = len(target_terms & evidence_terms) / max(len(target_terms), 1)
        relevance = min(1.0, overlap)
        if is_full:
            relevance = min(1.0, relevance + 0.15)
        evidence_item = FilteredEvidence(
            source_id=f"S{source_index}",
            url=res.url,
            title=page.title if page and page.title != "Untitled" else res.title,
            summary_or_excerpt=text_snippet,
            is_full_page=is_full,
            relevance_score=round(relevance, 3)
        )
        filtered.append(evidence_item)
        source_index += 1

    current_iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    domains = {urlparse(e.url).netloc.lower() for e in filtered}
    quality_sources = [e for e in filtered if e.relevance_score >= 0.5]
    covered_subquestions = sum(
        1 for subquestion in (plan.subquestions if plan else [])
        if any(
            len(_terms(subquestion) & _terms(e.summary_or_excerpt))
            >= max(2, len(_terms(subquestion)) // 3)
            for e in filtered
        )
    )
    planned_count = len(plan.subquestions) if plan else 0
    coverage_ok = planned_count == 0 or covered_subquestions >= max(1, (planned_count + 1) // 2)
    is_sufficient = (
        len(quality_sources) >= 2
        and len(domains) >= 2
        and coverage_ok
    ) or (
        len(quality_sources) >= 1
        and any(e.is_full_page and e.relevance_score >= 0.5 for e in filtered)
        and coverage_ok
    )

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
