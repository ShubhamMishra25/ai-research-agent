"""Domain exceptions and fallback builders for graceful failure handling."""
from typing import List
from app.schemas import ResearchResponse, ConfidenceAssessment, ConfidenceLevel, SourceReference, KeyFinding


class AgentBaseException(Exception):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


class ToolExecutionError(AgentBaseException):
    pass


class SSRFViolationError(ToolExecutionError):
    def __init__(self, message: str):
        super().__init__(message, recoverable=False)


def create_fallback_response(
    question: str,
    reason: str,
    sources: List[SourceReference] = None,
    errors: List[str] = None
) -> ResearchResponse:
    err_list = errors or []
    err_context = f" Reason: {reason}." if reason else ""
    return ResearchResponse(
        question=question,
        short_answer=f"Unable to complete research with high confidence.{err_context}",
        key_findings=[
            KeyFinding(
                finding="Execution encountered a bounded limit or failure mode preventing exhaustive analysis.",
                source_ids=[]
            )
        ],
        sources=sources or [],
        confidence=ConfidenceAssessment(
            level=ConfidenceLevel.LOW,
            justification=f"Deterministic fallback triggered due to errors or empty evidence: {reason}"
        ),
        assumptions=["Fallback response triggered without verified external facts."],
        limitations=["Tool outputs were either missing, malformed, or unrecoverable.", *err_list],
        suggested_next_steps=[
            "Verify external API keys (Tavily/Gemini).",
            "Broaden search parameters or provide direct source URLs.",
            "Inspect execution logs for transient network failures."
        ]
    )
