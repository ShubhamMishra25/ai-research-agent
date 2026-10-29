"""Planner node: produces structured research plans and targeted queries."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.schemas import ResearchPlan
from app.state import ResearchState
from app.utils.logging_config import setup_logger

logger = setup_logger("node.planner")

PLANNER_SYSTEM_PROMPT = """You are a senior research planning specialist.
Given a user question, generate a concise, highly targeted research plan.
Decompose the query into 2-3 specific subquestions and generate 2-3 precise web search queries.
Choose queries that will yield high-authority sources, benchmarks, official documentation, or industry overviews.
Avoid generic or overly broad search queries.
"""


def plan_research_node(state: ResearchState) -> dict:
    settings = get_settings()
    question = state["question"]
    logger.info(f"Generating research plan for query: '{question}'", extra={"execution_id": state.get("execution_id")})

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1
        )
        structured_llm = llm.with_structured_output(ResearchPlan)

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Research Query: {question}")
        ]
        plan: ResearchPlan = structured_llm.invoke(messages)

        return {
            "plan": plan,
            "iteration": state.get("iteration", 0) + 1,
            "warnings": []
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
            "iteration": state.get("iteration", 0) + 1,
            "warnings": [f"Planner fallback triggered: {str(exc)}"]
        }
