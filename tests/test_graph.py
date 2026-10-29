"""Tests end-to-end LangGraph state transitions and conditional logic."""
import pytest
from unittest.mock import patch, MagicMock
from app.graph import create_research_graph
from app.schemas import ResearchPlan, SearchResultItem, FetchedPage, ResearchResponse, ConfidenceAssessment, ConfidenceLevel, KeyFinding, SourceReference


def test_graph_happy_path_mocked():
    graph = create_research_graph()

    mock_plan = ResearchPlan(
        objective="Analyze vector databases",
        subquestions=["What are top vector DBs?"],
        proposed_searches=["vector databases comparison"],
        intended_tools=["web_search", "fetch_page"],
        evidence_needed="Performance benchmarks",
        completion_criteria="Direct comparison"
    )

    mock_response = ResearchResponse(
        question="Compare vector databases",
        short_answer="Qdrant, Chroma, and Milvus are leading choices.",
        key_findings=[KeyFinding(finding="Qdrant offers Rust core.", source_ids=["S1"])],
        sources=[SourceReference(source_id="S1", title="Vector DB", url="https://example.com/vdb", relevance_description="Comparison")],
        confidence=ConfidenceAssessment(level=ConfidenceLevel.HIGH, justification="Solid documentation"),
        assumptions=[],
        limitations=[],
        suggested_next_steps=[]
    )

    with patch("app.nodes.planner.ChatGoogleGenerativeAI.with_structured_output") as mock_planner_llm, \
         patch("app.tools.web_search.WebSearchTool.search") as mock_search, \
         patch("app.tools.fetch_page.PageFetchTool.fetch") as mock_fetch, \
         patch("app.nodes.synthesizer.ChatGoogleGenerativeAI.with_structured_output") as mock_synth_llm:

        mock_planner_llm.return_value.invoke.return_value = mock_plan
        mock_search.return_value = (
            [SearchResultItem(title="Vector DB", url="https://example.com/vdb", snippet="Comparison of Milvus and Qdrant")],
            MagicMock(status="SUCCESS", latency_ms=10.0, error_message=None)
        )
        mock_fetch.return_value = (
            FetchedPage(url="https://example.com/vdb", title="Vector DB", extracted_text="Comprehensive benchmark text", http_status=200, char_count=500),
            MagicMock(status="SUCCESS", latency_ms=25.0, error_message=None)
        )
        mock_synth_llm.return_value.invoke.return_value = mock_response

        state_input = {
            "execution_id": "test-run-1",
            "question": "Compare vector databases",
            "iteration": 0,
            "max_iterations": 2,
            "plan": None,
            "tool_records": [],
            "raw_search_results": [],
            "fetched_pages": [],
            "evidence": [],
            "is_sufficient": False,
            "sufficiency_reason": "Init",
            "final_response": None,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }

        output = graph.invoke(state_input)
        assert output["final_response"] is not None
        assert output["final_response"].short_answer.startswith("Qdrant, Chroma")
        assert len(output["evidence"]) >= 1
