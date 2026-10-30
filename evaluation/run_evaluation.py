"""Evaluation runner executing test cases and grading schema, grounding, facts, and confidence."""
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.graph import run_research_agent  # noqa: E402
from app.schemas import FetchedPage, ToolCallRecord  # noqa: E402
from app.schemas import ResearchResponse  # noqa: E402

console = Console()


def _failure_record(tool_name: str, target: str, message: str) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=tool_name,
        query_or_url=target,
        status="FAILURE",
        latency_ms=0.0,
        error_message=message,
    )


def _injected_search_failure(self, query: str, max_results: int = 4):
    return [], _failure_record("web_search", query, "Injected search outage")


def _injected_empty_search(self, query: str, max_results: int = 4):
    return [], ToolCallRecord(
        tool_name="web_search",
        query_or_url=query,
        status="SUCCESS",
        latency_ms=0.0,
    )


def _injected_fetch_failure(self, url: str):
    message = "Injected page-fetch outage"
    return (
        FetchedPage(
            url=url,
            title="Error Loading Page",
            extracted_text="",
            http_status=500,
            char_count=0,
            error=message,
        ),
        _failure_record("fetch_page", url, message),
    )


def run_case(tc: dict) -> tuple[dict, str]:
    """Run one case, applying deterministic fault injection when configured."""
    inject = tc.get("inject") or {}
    mode = "deterministic" if inject else "live"

    with contextlib.ExitStack() as stack:
        if inject.get("offline_planner"):
            stack.enter_context(
                patch(
                    "app.nodes.planner.ChatGoogleGenerativeAI",
                    side_effect=RuntimeError("Injected planner outage"),
                )
            )
        if inject.get("search_failure"):
            stack.enter_context(
                patch("app.tools.web_search.WebSearchTool.search", _injected_search_failure)
            )
        elif inject.get("empty_search"):
            stack.enter_context(
                patch("app.tools.web_search.WebSearchTool.search", _injected_empty_search)
            )
        if inject.get("fetch_failure"):
            stack.enter_context(
                patch("app.tools.fetch_page.PageFetchTool.fetch", _injected_fetch_failure)
            )
        return run_research_agent(tc["query"]), mode


def _answer_text(resp: ResearchResponse) -> str:
    parts = [resp.short_answer, *(finding.finding for finding in resp.key_findings)]
    return " ".join(parts).casefold()


def evaluate_facts(tc: dict, resp: ResearchResponse) -> dict:
    groups = tc.get("expected_entities", [])
    min_entities = tc.get("min_entities", len(groups))
    answer = _answer_text(resp)

    matched = []
    for group in groups:
        aliases = group if isinstance(group, list) else [group]
        if any(str(alias).casefold() in answer for alias in aliases):
            matched.append(str(aliases[0]))

    assumptions_ok = bool(resp.assumptions) if tc.get("require_assumptions") else True
    return {
        "expected_entity_count": len(groups),
        "entity_match_count": len(matched),
        "matched_entities": matched,
        "fact_level_matched": len(matched) >= min_entities,
        "assumptions_present": bool(resp.assumptions),
        "assumptions_requirement_matched": assumptions_ok,
    }


def evaluate_response(tc: dict, state: dict) -> dict:
    resp = state.get("final_response")

    results = {
        "id": tc["id"],
        "category": tc["category"],
        "schema_valid": False,
        "has_sources": len(resp.sources) > 0 if resp else False,
        "confidence_matched": False,
        "notes": "",
    }

    if not resp:
        results["notes"] = "Failed to produce final response schema"
        return results
    try:
        ResearchResponse.model_validate(resp)
        results["schema_valid"] = True
    except Exception as exc:
        results["notes"] = f"Invalid response schema: {exc}"
        return results

    actual_conf = resp.confidence.level.value
    expected_conf = tc.get("expected_confidence")
    results["confidence_matched"] = actual_conf.lower() == expected_conf.lower()
    results["actual_confidence"] = actual_conf

    cited_ids = set()
    for kf in resp.key_findings:
        cited_ids.update(kf.source_ids)

    evidence_by_id = {item.source_id: item for item in state.get("evidence", [])}
    declared_sources = {source.source_id: source for source in resp.sources}
    ids_match_evidence = cited_ids.issubset(evidence_by_id)
    urls_match_evidence = all(
        source_id in evidence_by_id
        and declared_sources[source_id].url == evidence_by_id[source_id].url
        for source_id in declared_sources
    )
    results["grounding_intact"] = ids_match_evidence and urls_match_evidence
    results["finding_citations_declared"] = cited_ids.issubset(set(declared_sources))
    if tc["expected_confidence"].lower() == "low":
        results["fallback_behavior_matched"] = (
            actual_conf.lower() == "low"
            and bool(resp.limitations)
            and "Unable to complete research" in resp.short_answer
        )
    else:
        results["fallback_behavior_matched"] = actual_conf.lower() != "low"

    results.update(evaluate_facts(tc, resp))

    metadata = state.get("metadata") or {}
    node_metrics = metadata.get("metrics") or []
    total_input_tokens = sum(
        (entry.get("usage") or {}).get("input_tokens") or 0 for entry in node_metrics
    )
    total_output_tokens = sum(
        (entry.get("usage") or {}).get("output_tokens") or 0 for entry in node_metrics
    )
    results["total_latency_ms"] = metadata.get("total_latency_ms")
    results["total_input_tokens"] = total_input_tokens
    results["total_output_tokens"] = total_output_tokens
    results["node_latencies_ms"] = {
        entry.get("node", "?"): entry.get("latency_ms") for entry in node_metrics
    }
    results["short_answer"] = resp.short_answer
    results["key_findings_count"] = len(resp.key_findings)
    results["sources_count"] = len(resp.sources)
    results["limitations"] = list(resp.limitations)
    results["assumptions"] = list(resp.assumptions)
    results["evidence_count"] = len(state.get("evidence", []))
    results["iteration_count"] = state.get("iteration", 0)
    results["errors"] = list(state.get("errors", []))
    results["warnings"] = list(state.get("warnings", []))
    results["overall_pass"] = all(
        [
            results["schema_valid"],
            results["grounding_intact"],
            results["confidence_matched"],
            results["fact_level_matched"],
            results["fallback_behavior_matched"],
            results["assumptions_requirement_matched"],
        ]
    )
    return results


def main():
    cases_path = Path(__file__).parent / "test_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    console.print(f"[bold cyan]Starting Evaluation Suite ({len(cases)} test cases)...[/bold cyan]\n")
    eval_summary = []

    for tc in cases:
        console.print(f"Executing [bold]{tc['id']}[/bold]: {tc['query']}")
        start = time.perf_counter()
        try:
            state, mode = run_case(tc)
            eval_res = evaluate_response(tc, state)
            eval_res["mode"] = mode
            eval_res["duration_sec"] = round(time.perf_counter() - start, 2)
            eval_summary.append(eval_res)
        except Exception as e:
            console.print(f"[red]Error on {tc['id']}: {str(e)}[/red]")
            eval_summary.append({
                "id": tc["id"],
                "category": tc["category"],
                "schema_valid": False,
                "has_sources": False,
                "confidence_matched": False,
                "grounding_intact": False,
                "fact_level_matched": False,
                "fallback_behavior_matched": False,
                "assumptions_requirement_matched": False,
                "overall_pass": False,
                "mode": "deterministic" if tc.get("inject") else "live",
                "duration_sec": round(time.perf_counter() - start, 2),
                "notes": str(e)
            })

    table = Table(title="Agent Benchmark Evaluation Results")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Schema", style="green")
    table.add_column("Grounding")
    table.add_column("Confidence")
    table.add_column("Facts")
    table.add_column("Fallback")
    table.add_column("Overall", style="bold")
    table.add_column("Duration (s)", justify="right")

    for item in eval_summary:
        table.add_row(
            item["id"],
            item["category"],
            "PASS" if item["schema_valid"] else "FAIL",
            "PASS" if item.get("grounding_intact") else "FAIL",
            "PASS" if item.get("confidence_matched") else "FAIL",
            "PASS" if item.get("fact_level_matched") else "FAIL",
            "PASS" if item.get("fallback_behavior_matched") else "FAIL",
            "PASS" if item.get("overall_pass") else "FAIL",
            str(item.get("duration_sec", 0.0)),
        )

    results_path = Path(__file__).parent / "evaluation_results.json"
    payload = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "case_count": len(cases),
            "gemini_model": get_settings().gemini_model,
            "max_research_iterations": get_settings().max_research_iterations,
            "max_retries": get_settings().max_retries,
            "llm_timeout_seconds": get_settings().llm_timeout_seconds,
            "live_external_services": True,
            "deterministic_fault_cases": [
                case["id"] for case in cases if case.get("inject")
            ],
        },
        "results": eval_summary,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\nMachine-readable results written to {results_path}")
    console.print(table)

    passed = sum(1 for item in eval_summary if item.get("overall_pass"))
    console.print(f"\n[bold]Overall: {passed}/{len(eval_summary)} cases passed[/bold]")


if __name__ == "__main__":
    main()
