"""CLI entrypoint for interactive and programmatic research runs."""
import argparse
import json
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import get_settings
from app.graph import run_research_agent
from app.utils.logging_config import setup_logger

console = Console()
logger = setup_logger("main")


def display_trace(result: dict) -> None:
    records = result.get("tool_records", [])
    if not records:
        return

    table = Table(title="Agent Tool Execution Trace")
    table.add_column("Tool", style="cyan")
    table.add_column("Target / Query", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Attempts", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for rec in records:
        status_color = "green" if rec.status == "SUCCESS" else "red"
        table.add_row(
            rec.tool_name,
            rec.query_or_url[:60] + "..." if len(rec.query_or_url) > 60 else rec.query_or_url,
            f"[{status_color}]{rec.status}[/{status_color}]",
            str(getattr(rec, "attempts", 1)),
            f"{rec.latency_ms:.1f}"
        )
    console.print(table)


def display_metrics(result: dict) -> None:
    metadata = result.get("metadata") or {}
    metrics = metadata.get("metrics") or []
    if not metrics:
        return

    table = Table(title="Agent Node Metrics")
    table.add_column("Node", style="cyan")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Details", style="magenta")

    for entry in metrics:
        details = {
            key: value
            for key, value in entry.items()
            if key not in ("node", "latency_ms") and value not in (None, {}, [], "")
        }
        table.add_row(entry.get("node", "?"), f"{entry.get('latency_ms', 0):.1f}", str(details))
    console.print(table)
    if metadata.get("total_latency_ms") is not None:
        console.print(f"[dim]Total workflow latency: {metadata['total_latency_ms']:.1f} ms[/dim]")


def main():
    parser = argparse.ArgumentParser(description="AI Production Research Agent CLI")
    parser.add_argument("query", nargs="?", type=str, help="Research question to investigate")
    parser.add_argument("--json", action="store_true", help="Print only raw JSON output")
    parser.add_argument("--trace", action="store_true", help="Print tool execution trace table")
    args = parser.parse_args()

    query = args.query
    if not query:
        console.print("[bold yellow]Enter research question:[/bold yellow] ", end="")
        query = input().strip()

    if not query:
        console.print("[bold red]Error: Query cannot be empty.[/bold red]")
        sys.exit(1)

    settings = get_settings()
    if not settings.gemini_api_key:
        console.print("[bold red]Error: GEMINI_API_KEY environment variable is not set.[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Initiating research:[/bold cyan] {query}")

    try:
        final_state = run_research_agent(query)
        final_resp = final_state.get("final_response")

        if args.trace:
            display_trace(final_state)
            display_metrics(final_state)

        if not final_resp:
            console.print("[bold red]Research terminated without generating a response.[/bold red]")
            sys.exit(2)

        resp_dict = final_resp.model_dump()

        if args.json:
            print(json.dumps(resp_dict, indent=2))
        else:
            console.print("\n")
            console.print(Panel(
                f"[bold green]Short Answer:[/bold green]\n{resp_dict['short_answer']}\n\n"
                f"[bold cyan]Confidence:[/bold cyan] {resp_dict['confidence']['level']} "
                f"({resp_dict['confidence']['justification']})",
                title=f"Research Output: {resp_dict['question']}",
                border_style="blue"
            ))

            console.print("\n[bold]Key Findings:[/bold]")
            for kf in resp_dict["key_findings"]:
                sources_cited = f" [cyan]Sources: {', '.join(kf['source_ids'])}[/cyan]" if kf["source_ids"] else ""
                console.print(f"  • {kf['finding']}{sources_cited}")

            console.print("\n[bold]Sources Consulted:[/bold]")
            for src in resp_dict["sources"]:
                console.print(f"  [{src['source_id']}] [link={src['url']}]{src['title']}[/link] - {src['url']}")

            console.print("\n[bold]Assumptions:[/bold]")
            for assumption in resp_dict["assumptions"]:
                console.print(f"  • {assumption}")

            console.print("\n[bold]Limitations:[/bold]")
            for lim in resp_dict["limitations"]:
                console.print(f"  • {lim}")

            console.print("\n[bold]Suggested Next Steps:[/bold]")
            for step in resp_dict["suggested_next_steps"]:
                console.print(f"  ➜ {step}")

    except Exception as exc:
        logger.error(f"Execution failed: {str(exc)}", exc_info=True)
        console.print(f"[bold red]Fatal error:[/bold red] {str(exc)}")
        sys.exit(3)


if __name__ == "__main__":
    main()
