"""Planner node: produces structured research plans and targeted queries."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.schemas import FollowUpSearchPlan, ResearchPlan
from app.state import ResearchState
from app.utils.llm import invoke_structured
from app.utils.logging_config import setup_logger
from app.utils.metrics import track_node

logger = setup_logger("node.planner")

PLANNER_SYSTEM_PROMPT = """You are a senior research planning specialist.
Given a user question, generate a concise, highly targeted research plan.
Decompose the query into 2-3 specific subquestions and generate 2-3 precise web search queries.
Choose queries that will yield high-authority sources, benchmarks, official documentation, or industry overviews.
Avoid generic or overly broad search queries.
For latest, new, recent, or current questions, include recency-aware queries.
For organization-specific questions, prefer that organization's official
announcement and documentation domains.
"""

FOLLOW_UP_SYSTEM_PROMPT = """You are a research refinement specialist.
Generate 1-2 precise, non-overlapping web search queries that directly address
the supplied unanswered subquestions. Prefer authoritative sources and include
the missing aspect in each query. Do not repeat any query already used.
For temporal gaps, include the relevant year or "latest official announcement"
when appropriate.
"""


@track_node("planner")
def plan_research_node(state: ResearchState) -> dict:
    settings = get_settings()
    question = state["question"]
    logger.info(f"Generating research plan for query: '{question}'", extra={"execution_id": state.get("execution_id")})

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Research Query: {question}")
        ]
        plan, usage = invoke_structured(
            llm,
            ResearchPlan,
            messages,
            settings=settings,
            description="planner structured output",
        )

        return {
            "plan": plan,
            "warnings": list(state.get("warnings", [])),
            "_metrics": {"usage": usage},
        }
    except Exception as exc:
        logger.warning(f"Planner LLM failed, using deterministic fallback plan: {str(exc)}")
        fallback_plan = ResearchPlan(
            objective=f"Analyze and answer: {question}",
            subquestions=[question],
            proposed_searches=[question, f"{question} comparison overview"],
            intended_tools=["web_search", "fetch_page"],
            evidence_needed="Factual benchmarks, primary documentation, and verifiable summaries.",
            completion_criteria="Direct answers found for core user queries."
        )
        return {
            "plan": fallback_plan,
            "warnings": [
                *state.get("warnings", []),
                f"Planner fallback triggered: {str(exc)}",
            ]
        }


def generate_follow_up_queries(state: ResearchState) -> list[str]:
    """Generate targeted searches for current evidence gaps.

    A deterministic fallback keeps the research loop useful when the planning
    model is unavailable.
    """
    gaps = [gap.strip() for gap in state.get("evidence_gaps", []) if gap.strip()]
    used = {query.casefold().strip() for query in state.get("used_queries", [])}

    if not gaps:
        # Insufficient evidence without a per-subquestion gap (e.g. too few
        # independent domains). Fall back to the planned subquestions so the
        # extra research cycle still does useful work.
        plan = state.get("plan")
        if plan and plan.subquestions:
            gaps = [subquestion.strip() for subquestion in plan.subquestions if subquestion.strip()]
        if not gaps:
            gaps = [state.get("question", "").strip()]
        gaps = [gap for gap in gaps if gap]

    if not gaps:
        return []

    try:
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        prompt = (
            f"Research question: {state['question']}\n"
            f"Evidence gaps: {gaps}\n"
            f"Previously used queries: {list(state.get('used_queries', []))}"
        )
        result, _usage = invoke_structured(
            llm,
            FollowUpSearchPlan,
            [
                SystemMessage(content=FOLLOW_UP_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            settings=settings,
            description="follow-up query generation",
        )
        candidates = result.queries
    except Exception as exc:
        logger.warning("Follow-up query generation failed: %s", exc)
        candidates = [f"{gap} authoritative evidence" for gap in gaps]

    queries = []
    for query in candidates:
        normalized = query.strip()
        if normalized and normalized.casefold() not in used and normalized.casefold() not in {
            item.casefold() for item in queries
        }:
            queries.append(normalized)
    return queries[:2]
