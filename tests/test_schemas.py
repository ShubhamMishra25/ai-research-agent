"""Tests schema validation, serialization, and boundary conditions."""
import pytest
from pydantic import ValidationError
from app.schemas import ResearchResponse, KeyFinding, SourceReference, ConfidenceAssessment, ConfidenceLevel


def test_research_response_valid():
    response = ResearchResponse(
        question="Compare Milvus and Qdrant",
        short_answer="Both are high performance vector databases.",
        key_findings=[
            KeyFinding(finding="Qdrant is written in Rust.", source_ids=["S1"])
        ],
        sources=[
            SourceReference(source_id="S1", title="Qdrant Docs", url="https://qdrant.tech", relevance_description="Primary docs")
        ],
        confidence=ConfidenceAssessment(level=ConfidenceLevel.HIGH, justification="Direct primary documentation"),
        assumptions=["Evaluated for standard vector search workloads"],
        limitations=["Benchmarks depend on specific embedding dimensions"],
        suggested_next_steps=["Run local latency benchmarks on customer dataset"]
    )
    assert response.confidence.level == ConfidenceLevel.HIGH
    assert len(response.sources) == 1
    assert response.key_findings[0].source_ids == ["S1"]


def test_research_response_missing_fields_raises():
    with pytest.raises(ValidationError):
        ResearchResponse(
            question="What is LangGraph?",
            short_answer="Orchestration framework.",
            key_findings=[]
        )
