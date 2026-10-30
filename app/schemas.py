"""Pydantic schemas for structured inputs, plans, and outputs."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ToolCallRecord(BaseModel):
    tool_name: str
    query_or_url: str
    status: str  # "SUCCESS" or "FAILURE"
    latency_ms: float
    attempts: int = 1
    error_message: Optional[str] = None


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str
    score: Optional[float] = 0.0


class FetchedPage(BaseModel):
    url: str
    canonical_url: Optional[str] = None
    title: str
    extracted_text: str
    http_status: int
    char_count: int
    error: Optional[str] = None


class FilteredEvidence(BaseModel):
    source_id: str  # e.g., "S1", "S2"
    url: str
    title: str
    summary_or_excerpt: str
    is_full_page: bool = False
    relevance_score: float = 1.0


class ResearchPlan(BaseModel):
    objective: str = Field(description="Clear restatement of research target")
    subquestions: List[str] = Field(description="Logical subquestions that decompose the query")
    proposed_searches: List[str] = Field(description="Specific targeted queries for search tool")
    intended_tools: List[str] = Field(default=["web_search", "fetch_page"], description="Tools selected")
    evidence_needed: str = Field(description="What evidence must be gathered to answer fully")
    completion_criteria: str = Field(description="Condition indicating research is complete")


class FollowUpSearchPlan(BaseModel):
    queries: List[str] = Field(
        description="New search queries targeting the supplied evidence gaps"
    )


class KeyFinding(BaseModel):
    finding: str = Field(description="Key factual finding or takeaway")
    source_ids: List[str] = Field(default_factory=list, description="IDs of sources supporting this finding (e.g. S1)")


class SourceReference(BaseModel):
    source_id: str = Field(description="Unique source ID, e.g., S1")
    title: str = Field(description="Title of page or search result")
    url: str = Field(description="Full URL of the source")
    relevance_description: str = Field(description="Brief explanation of how this source was used")


class ConfidenceAssessment(BaseModel):
    level: ConfidenceLevel
    justification: str = Field(description="Rationale based on evidence completeness and corroboration")


class ResearchResponse(BaseModel):
    question: str
    short_answer: str = Field(description="Concise direct answer to user question")
    key_findings: List[KeyFinding] = Field(description="Structured evidence-backed findings")
    sources: List[SourceReference] = Field(description="List of cited sources with source_ids")
    confidence: ConfidenceAssessment
    assumptions: List[str] = Field(description="Assumptions made during analysis")
    limitations: List[str] = Field(description="Known gaps, data recency limits, or missing evidence")
    suggested_next_steps: List[str] = Field(description="Actionable follow-up steps for the user")
