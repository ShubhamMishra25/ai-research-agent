"""Tests resilience when external tools fail completely."""
from unittest.mock import patch, MagicMock
from app.graph import create_research_graph
from app.schemas import ConfidenceLevel


def test_graph_complete_search_failure_produces_fallback():
    graph = create_research_graph()

    with patch("app.tools.web_search.WebSearchTool.search") as mock_search:
        mock_search.return_value = (
            [],
            MagicMock(status="FAILURE", latency_ms=50.0, error_message="API connection timed out")
        )

        state_input = {
            "execution_id": "failure-test-1",
            "question": "Research nonexistent concept",
            "iteration": 0,
            "max_iterations": 1,
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
        final_resp = output["final_response"]
        assert final_resp is not None
        assert final_resp.confidence.level == ConfidenceLevel.LOW
        assert "Unable to complete research" in final_resp.short_answer
