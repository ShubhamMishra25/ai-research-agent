"""LangGraph research agent assembly with conditional routing and bounded iteration."""
import time
from uuid import uuid4
from langgraph.graph import StateGraph, START, END

from app.config import get_settings
from app.state import ResearchState
from app.nodes.planner import plan_research_node
from app.nodes.researcher import execute_research_node
from app.nodes.evidence import process_evidence_node
from app.nodes.synthesizer import synthesize_answer_node
from app.utils.logging_config import setup_logger
from app.utils.metrics import utc_now_iso

logger = setup_logger("graph")


def should_continue_research(state: ResearchState) -> str:
    is_sufficient = state.get("is_sufficient", False)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", get_settings().max_research_iterations)

    if is_sufficient or iteration >= max_iterations:
        return "synthesize"
    logger.info(f"Evidence insufficient. Routing for research iteration {iteration + 1}")
    return "research"


def create_research_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", plan_research_node)
    workflow.add_node("researcher", execute_research_node)
    workflow.add_node("evidence", process_evidence_node)
    workflow.add_node("synthesizer", synthesize_answer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "evidence")

    workflow.add_conditional_edges(
        "evidence",
        should_continue_research,
        {
            "research": "researcher",
            "synthesize": "synthesizer"
        }
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


def run_research_agent(question: str, execution_id: str = None) -> ResearchState:
    app = create_research_graph()
    settings = get_settings()
    started_at = time.perf_counter()

    initial_state: ResearchState = {
        "execution_id": execution_id or str(uuid4()),
        "question": question.strip(),
        "iteration": 0,
        "max_iterations": settings.max_research_iterations,
        "plan": None,
        "tool_records": [],
        "raw_search_results": [],
        "fetched_pages": [],
        "evidence": [],
        "evidence_gaps": [],
        "used_queries": [],
        "is_sufficient": False,
        "sufficiency_reason": "Initialized",
        "final_response": None,
        "errors": [],
        "warnings": [],
        "metadata": {"started_at": utc_now_iso()}
    }

    result = app.invoke(initial_state)
    metadata = result.get("metadata") or {}
    metadata["completed_at"] = utc_now_iso()
    metadata["total_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    result["metadata"] = metadata
    return result
