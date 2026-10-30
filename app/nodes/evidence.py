"""Evidence processing node: filters, deduplicates, and assesses sufficiency."""
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.schemas import FilteredEvidence
from app.state import ResearchState
from app.utils.logging_config import setup_logger
from app.utils.metrics import track_node
from app.utils.urls import canonicalize_url

logger = setup_logger("node.evidence")

STOPWORDS = {
    "about", "after", "also", "and", "are", "best", "but", "can", "could",
    "did", "does", "find", "for", "from", "has", "have", "how", "if", "into",
    "its", "may", "not", "our", "please", "should", "than", "that", "the",
    "their", "them", "then", "these", "they", "this", "use", "used", "using",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}
TEMPORAL_TERMS = {"latest", "new", "recent", "current", "today", "now", "recently"}
OFFICIAL_DOMAINS = {
    "openai": {"openai.com", "platform.openai.com", "help.openai.com"},
    "google": {"google.com", "deepmind.google", "ai.google"},
    "microsoft": {"microsoft.com", "blogs.microsoft.com"},
    "anthropic": {"anthropic.com"},
    "meta": {"meta.com", "ai.meta.com"},
}
NON_MODEL_TERMS = {"chip", "hardware", "inference", "datacenter", "infrastructure"}


def _stem(term: str) -> str:
    """Very small, deterministic stemmer for lexical matching.

    It is intentionally conservative: it only normalizes common English
    plurals and suffixes so that "database"/"databases" and
    "framework"/"frameworks" compare equal. It is not a language model.
    """
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _terms(text: str) -> set[str]:
    return {
        _stem(term)
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in STOPWORDS
    }


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _temporal_query(question: str) -> bool:
    return bool(_terms(question) & TEMPORAL_TERMS)


def _organization_domains(question: str) -> set[str]:
    question_lower = question.lower()
    domains = set()
    for organization, candidates in OFFICIAL_DOMAINS.items():
        if organization in question_lower:
            domains.update(candidates)
    return domains


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value)
    if not match:
        match = re.search(r"\b(20\d{2})[-/](\d{1,2})\b", value)
        if match:
            try:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    1,
                    tzinfo=timezone.utc,
                )
            except ValueError:
                return None
        match = re.search(r"\b(20\d{2})\b", value)
        if not match:
            return None
        return datetime(int(match.group(1)), 1, 1, tzinfo=timezone.utc)
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        # Search providers occasionally return malformed publication dates.
        # Treat those as unknown rather than allowing one result to abort the run.
        return None


def _recency_score(published_date: str | None, text: str, temporal: bool) -> float:
    if not temporal:
        return 1.0
    date = _parse_date(published_date) or _parse_date(text)
    if not date:
        return 0.2
    age_days = max(0, (datetime.now(timezone.utc) - date).days)
    if age_days <= 365:
        return 1.0
    if age_days <= 730:
        return 0.5
    return 0.1


def _authority_score(url: str, preferred_domains: set[str], text: str) -> float:
    domain = _domain(url)
    if domain in preferred_domains:
        return 1.0
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.85
    if "official" in text.lower() or "documentation" in text.lower():
        return 0.65
    return 0.35


@track_node("evidence")
def process_evidence_node(state: ResearchState) -> dict:
    raw_results = state.get("raw_search_results", [])
    fetched_pages = {p.url: p for p in state.get("fetched_pages", []) if p.char_count > 50}

    filtered: list[FilteredEvidence] = []
    question_terms = _terms(state.get("question", ""))
    plan = state.get("plan")
    planned_terms = _terms(" ".join(plan.subquestions)) if plan else set()
    target_terms = question_terms | planned_terms
    temporal = _temporal_query(state.get("question", ""))
    preferred_domains = _organization_domains(state.get("question", ""))
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
        # Only deduplicate identical content from the same domain. The same
        # text on two different domains may be two independent (or
        # syndicated) sources; discarding the second loses corroboration.
        content_key = (_domain(res.url), normalized_snippet.casefold()[:300])
        if len(content_key[1]) >= 20 and content_key in seen_content:
            duplicate_count += 1
            continue

        evidence_terms = _terms(f"{res.title} {res.snippet} {text_snippet}")
        overlap = len(target_terms & evidence_terms) / max(len(target_terms), 1)

        # Take the stronger of lexical overlap and the search engine's own
        # ranking signal. Averaging them suppressed highly ranked results for
        # broad/entity-discovery queries whose snippets do not repeat every
        # generic query term. Snippets with no score still work (direct URLs).
        relevance = overlap
        if res.score and res.score > 0:
            relevance = max(overlap, min(res.score, 1.0))
        relevance = min(1.0, relevance)
        if is_full:
            relevance = min(1.0, relevance + 0.15)
        combined_text = f"{res.title} {res.snippet} {text_snippet}"
        authority = _authority_score(res.url, preferred_domains, combined_text)
        recency = _recency_score(res.published_date, combined_text, temporal)
        # Hardware/infrastructure results should not qualify as model-release
        # evidence merely because they mention the organization.
        if "model" in target_terms and _terms(combined_text) & NON_MODEL_TERMS:
            relevance = min(relevance, 0.35)

        seen_urls.add(url_key)
        if len(content_key[1]) >= 20:
            seen_content.add(content_key)

        evidence_item = FilteredEvidence(
            source_id=f"S{source_index}",
            url=res.url,
            title=page.title if page and page.title != "Untitled" else res.title,
            summary_or_excerpt=text_snippet,
            is_full_page=is_full,
            relevance_score=round(relevance, 3)
            ,
            authority_score=authority,
            recency_score=recency,
            published_date=res.published_date,
        )
        filtered.append(evidence_item)
        source_index += 1

    current_iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    domains = {_domain(e.url) for e in filtered}
    quality_sources = [
        e for e in filtered
        if e.relevance_score >= 0.5
        and e.authority_score >= (0.6 if preferred_domains else 0.35)
        and e.recency_score >= (0.5 if temporal else 0.0)
    ]
    covered_subquestions = sum(
        1 for subquestion in (plan.subquestions if plan else [])
        if any(
            len(_terms(subquestion) & _terms(e.summary_or_excerpt))
            >= max(2, len(_terms(subquestion)) // 3)
            for e in filtered
        )
    )
    planned_count = len(plan.subquestions) if plan else 0
    # Every planned subquestion must be covered before we stop early.
    # Partial coverage is exactly what the follow-up loop exists to fix.
    coverage_ok = planned_count == 0 or covered_subquestions == planned_count
    # A single relevant source is not enough to stop research: it cannot
    # corroborate itself and often misses requested subquestions. Require at
    # least two quality sources across two domains, or keep iterating until
    # the bounded budget is exhausted.
    is_sufficient = (
        len(quality_sources) >= 2
        and len(domains) >= 2
        and coverage_ok
        and (not preferred_domains or any(e.authority_score >= 0.9 for e in quality_sources))
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
            "covered_subquestions": covered_subquestions,
            "planned_subquestions": planned_count,
            "evidence_gaps": len(evidence_gaps),
            "temporal_query": temporal,
            "preferred_domains": sorted(preferred_domains),
        },
    }
