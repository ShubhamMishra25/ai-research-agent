"""State schema definition for the LangGraph research workflow."""
from typing import TypedDict, List, Dict, Any, Optional
from app.schemas import (
    ResearchPlan,
    SearchResultItem,
    FetchedPage,
    FilteredEvidence,
    ToolCallRecord,
    ResearchResponse,
)


class ResearchState(TypedDict):
    execution_id: str
    question: str
    iteration: int
    max_iterations: int
    plan: Optional[ResearchPlan]
    tool_records: List[ToolCallRecord]
    raw_search_results: List[SearchResultItem]
    fetched_pages: List[FetchedPage]
    evidence: List[FilteredEvidence]
    evidence_gaps: List[str]
    used_queries: List[str]
    is_sufficient: bool
    sufficiency_reason: str
    final_response: Optional[ResearchResponse]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
