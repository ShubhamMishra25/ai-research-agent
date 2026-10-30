"""Tests end-to-end LangGraph state transitions and conditional logic."""
import pytest
from unittest.mock import patch, MagicMock
from app.graph import create_research_graph
from app.nodes.researcher import execute_research_node
from app.nodes.evidence import process_evidence_node
from app.schemas import (
    ResearchPlan,
    SearchResultItem,
    FetchedPage,
    ResearchResponse,
    ConfidenceAssessment,
    ConfidenceLevel,
    KeyFinding,
    SourceReference,
)


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

    # Use side_effect to route structured outputs based on the schema requested
    def structured_output_side_effect(schema):
        mock_runnable = MagicMock()
        if schema == ResearchPlan:
            mock_runnable.invoke.return_value = mock_plan
        else:
            mock_runnable.invoke.return_value = mock_response
        return mock_runnable

    with patch("langchain_google_genai.ChatGoogleGenerativeAI.with_structured_output", side_effect=structured_output_side_effect), \
         patch("app.tools.web_search.WebSearchTool.search") as mock_search, \
         patch("app.tools.fetch_page.PageFetchTool.fetch") as mock_fetch:

        mock_search.return_value = (
            [SearchResultItem(title="Vector DB", url="https://example.com/vdb", snippet="Comparison of Milvus and Qdrant")],
            MagicMock(status="SUCCESS", latency_ms=10.0, error_message=None)
        )
        mock_fetch.return_value = (
            FetchedPage(url="https://example.com/vdb", title="Vector DB", extracted_text="Comprehensive benchmark text", http_status=200, char_count=500),
            MagicMock(status="SUCCESS", latency_ms=25.0, error_message=None)
        )

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


def test_insufficient_evidence_runs_second_research_cycle():
    graph = create_research_graph()
    mock_plan = ResearchPlan(
        objective="Find evidence",
        subquestions=["Specific evidence"],
        proposed_searches=["specific evidence"],
        evidence_needed="Direct source",
        completion_criteria="One verified source",
    )
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.with_structured_output") as structured, \
         patch("app.tools.web_search.WebSearchTool.search") as search, \
         patch("app.tools.fetch_page.PageFetchTool.fetch") as fetch:
        planner_output = MagicMock()
        planner_output.invoke.return_value = mock_plan
        structured.return_value = planner_output
        search.return_value = ([], MagicMock(status="FAILURE", error_message="offline"))
        fetch.return_value = (None, MagicMock(status="FAILURE", error_message="offline"))
        state = {
            "execution_id": "iteration-test",
            "question": "Find evidence",
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
            "metadata": {},
        }
        output = graph.invoke(state)

    assert search.call_count == 2
    assert output["iteration"] == 2


def test_follow_up_queries_target_gaps_and_preserve_unique_evidence():
    state = {
        "execution_id": "adaptive-test",
        "question": "Compare vector databases",
        "iteration": 1,
        "max_iterations": 3,
        "plan": ResearchPlan(
            objective="Compare databases",
            subquestions=["Benchmark latency", "Compare licensing"],
            proposed_searches=["vector database overview"],
            evidence_needed="Benchmarks and licenses",
            completion_criteria="Both gaps answered",
        ),
        "evidence_gaps": ["Compare licensing"],
        "used_queries": ["vector database overview"],
        "tool_records": [],
        "raw_search_results": [
            SearchResultItem(
                title="Existing",
                url="https://example.com/existing",
                snippet="Existing benchmark evidence",
            )
        ],
        "fetched_pages": [],
        "evidence": [],
        "is_sufficient": False,
        "sufficiency_reason": "Missing licensing",
        "final_response": None,
        "errors": [],
        "warnings": [],
        "metadata": {},
    }

    with patch(
        "app.nodes.researcher.generate_follow_up_queries",
        return_value=["vector database overview", "vector database licensing comparison"],
    ), patch("app.tools.web_search.WebSearchTool.search") as search, patch(
        "app.tools.fetch_page.PageFetchTool.fetch"
    ) as fetch:
        search.return_value = (
            [
                SearchResultItem(
                    title="Existing duplicate",
                    url="https://example.com/existing",
                    snippet="Duplicate result",
                ),
                SearchResultItem(
                    title="License guide",
                    url="https://example.com/license",
                    snippet="Open-source licensing comparison",
                ),
            ],
            MagicMock(status="SUCCESS", latency_ms=1.0, error_message=None),
        )
        def fetch_page(url):
            return (
                FetchedPage(
                    url=url,
                    title="License guide",
                    extracted_text="Open-source licensing comparison",
                    http_status=200,
                    char_count=100,
                ),
                MagicMock(status="SUCCESS", latency_ms=1.0, error_message=None),
            )

        fetch.side_effect = fetch_page
        result = execute_research_node(state)

    assert search.call_args.args[0] == "vector database licensing comparison"
    assert result["used_queries"] == [
        "vector database overview",
        "vector database licensing comparison",
    ]
    assert [item.url for item in result["raw_search_results"]] == [
        "https://example.com/existing",
        "https://example.com/license",
    ]
    assert [page.url for page in result["fetched_pages"]] == [
        "https://example.com/existing",
        "https://example.com/license",
    ]


def test_sufficient_evidence_stops_research():
    plan = ResearchPlan(
        objective="Compare databases",
        subquestions=["Benchmark latency"],
        proposed_searches=["vector database benchmark"],
        evidence_needed="Latency benchmarks",
        completion_criteria="Benchmark covered",
    )
    result = process_evidence_node({
        "question": "Compare vector databases",
        "plan": plan,
        "iteration": 1,
        "max_iterations": 3,
        "raw_search_results": [
            SearchResultItem(
                title="Benchmark A",
                url="https://a.example/bench",
                snippet="Vector database benchmark latency results",
            ),
            SearchResultItem(
                title="Benchmark B",
                url="https://b.example/bench",
                snippet="Vector database benchmark latency comparison",
            ),
        ],
        "fetched_pages": [],
    })

    assert result["is_sufficient"] is True
    assert result["evidence_gaps"] == []


def test_follow_up_query_generation_failure_uses_gap_fallback():
    from app.nodes.planner import generate_follow_up_queries

    state = {
        "question": "Compare vector databases",
        "evidence_gaps": ["licensing differences"],
        "used_queries": ["vector database overview"],
    }
    with patch(
        "app.nodes.planner.ChatGoogleGenerativeAI",
        side_effect=RuntimeError("planner unavailable"),
    ):
        queries = generate_follow_up_queries(state)

    assert queries == ["licensing differences authoritative evidence"]