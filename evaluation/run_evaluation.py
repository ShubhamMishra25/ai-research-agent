"""Evaluation runner executing test cases and grading schema, grounding, and confidence."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.graph import run_research_agent

console = Console()


def evaluate_response(tc: dict, state: dict) -> dict:
    resp = state.get("final_response")

    results = {
        "id": tc["id"],
        "category": tc["category"],
        "schema_valid": resp is not None,
        "has_sources": len(resp.sources) > 0 if resp else False,
        "confidence_matched": False,
        "notes": ""
    }

    if not resp:
        results["notes"] = "Failed to produce final response schema"
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
            state = run_research_agent(tc["query"])
            eval_res = evaluate_response(tc, state)
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
                "duration_sec": round(time.perf_counter() - start, 2),
                "notes": str(e)
            })

    table = Table(title="Agent Benchmark Evaluation Results")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Schema Valid", style="green")
    table.add_column("Sources Present")
    table.add_column("Grounding Intact")
    table.add_column("Duration (s)", justify="right")

    for item in eval_summary:
        table.add_row(
            item["id"],
            item["category"],
            "✓" if item["schema_valid"] else "✗",
            "✓" if item.get("has_sources") else "✗",
            "✓" if item.get("grounding_intact") else "✗",
            str(item.get("duration_sec", 0.0))
        )

    results_path = Path(__file__).parent / "evaluation_results.json"
    payload = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "case_count": len(cases),
            "max_research_iterations": get_settings().max_research_iterations,
            "live_external_services": True,
        },
        "results": eval_summary,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\nMachine-readable results written to {results_path}")
    console.print(table)


if __name__ == "__main__":
    main()
