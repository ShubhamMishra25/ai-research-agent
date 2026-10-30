"""Evidence processing node: filters, deduplicates, and assesses sufficiency."""
import re
from urllib.parse import urlparse

from app.schemas import FilteredEvidence
from app.state import ResearchState
from app.utils.logging_config import setup_logger
from app.utils.metrics import track_node
from app.utils.urls import canonicalize_url

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


@track_node("evidence")
def process_evidence_node(state: ResearchState) -> dict:
    raw_results = state.get("raw_search_results", [])
    fetched_pages = {p.url: p for p in state.get("fetched_pages", []) if p.char_count > 50}

    filtered: list[FilteredEvidence] = []
    question_terms = _terms(state.get("question", ""))
    plan = state.get("plan")
    planned_terms = _terms(" ".join(plan.subquestions)) if plan else set()
    target_terms = question_terms | planned_terms
    source_index = 1
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    duplicate_count = 0

    for res in raw_results:
        url_key = canonicalize_url(res.url)
        if url_key in seen_urls:
            duplicate_count += 1
            continue

        page = fetched_pages.get(res.url)
        if page:
            text_snippet = page.extracted_text[:1200]
            is_full = True
        else:
            text_snippet = res.snippet
            is_full = False

        if not text_snippet or len(text_snippet.strip()) < 15:
            continue

        normalized_snippet = " ".join(text_snippet.split())
        content_key = normalized_snippet.casefold()[:300]
        if len(content_key) >= 20 and content_key in seen_content:
            duplicate_count += 1
            continue

        evidence_terms = _terms(f"{res.title} {res.snippet} {text_snippet}")
        overlap = len(target_terms & evidence_terms) / max(len(target_terms), 1)

        # Blend lexical relevance with the search engine's own ranking signal
        # when available. Snippets with no score still work (e.g. direct URLs).
        relevance = overlap
        if res.score and res.score > 0:
            relevance = 0.8 * overlap + 0.2 * min(res.score, 1.0)
        relevance = min(1.0, relevance)
        if is_full:
            relevance = min(1.0, relevance + 0.15)

        seen_urls.add(url_key)
        if len(content_key) >= 20:
            seen_content.add(content_key)

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
    evidence_gaps = []
    if plan:
        for subquestion in plan.subquestions:
            subquestion_terms = _terms(subquestion)
            covered = any(
                len(subquestion_terms & _terms(item.summary_or_excerpt))
                >= max(2, len(subquestion_terms) // 3)
                for item in filtered
            )
            if not covered:
                evidence_gaps.append(subquestion)

    if not is_sufficient:
        if current_iteration < max_iterations:
            reason = f"Insufficient evidence gathered ({len(filtered)} valid sources). Triggering refinement."
        else:
            reason = f"Budget exhausted after {current_iteration} iterations with limited evidence."
    else:
        reason = f"Sufficient evidence gathered ({len(filtered)} validated sources)."

    logger.info(
        "Evidence status: %s (raw=%d, kept=%d, duplicates=%d, domains=%d, quality=%d)",
        reason,
        len(raw_results),
        len(filtered),
        duplicate_count,
        len(domains),
        len(quality_sources),
        extra={"execution_id": state.get("execution_id")},
    )

    return {
        "evidence": filtered,
        "evidence_gaps": evidence_gaps,
        "is_sufficient": is_sufficient,
        "sufficiency_reason": reason,
        "_metrics": {
            "raw_results": len(raw_results),
            "evidence_kept": len(filtered),
            "duplicates_filtered": duplicate_count,
            "domains": len(domains),
            "quality_sources": len(quality_sources),
            "evidence_gaps": len(evidence_gaps),
        },
    }
