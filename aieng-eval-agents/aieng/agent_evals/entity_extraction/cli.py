#!/usr/bin/env python3
"""Entity Extraction Agent CLI.

Command-line interface for running the entity extraction agent against article data.

Usage::

    entity-extract run --file data/transformed_data/2017_data.csv --rows 3
    entity-extract run --file data/transformed_data/2017_data.csv --row-index 0
    entity-extract run --title "..." --maintext "..."
"""

import argparse
import asyncio
import csv
import logging
import sys
from importlib.metadata import version
from pathlib import Path

from aieng.agent_evals.configs import Configs
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from aieng.agent_evals.entity_extraction.agent import run_entity_extraction

console = Console()
VECTOR_CYAN = "#00B4D8"


def _load_env() -> None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return
    load_dotenv()


_load_env()


def get_version() -> str:
    try:
        return version("aieng-eval-agents")
    except Exception:
        return "dev"


def display_banner() -> None:
    worker_model = _get_model()
    ver = get_version()

    line0 = Text()
    line0.append("  ◯─◯    ", style=f"{VECTOR_CYAN} bold")
    line0.append("entity-extract ", style="white bold")
    line0.append(f"v{ver}", style="bright_black")

    line1 = Text()
    line1.append(" ╱ 🏷  ╲   ", style=f"{VECTOR_CYAN} bold")
    line1.append("Model: ", style="dim")
    line1.append(worker_model, style="cyan")

    line2 = Text()
    line2.append("  ╲__╱   ", style=f"{VECTOR_CYAN} bold")
    line2.append("Vector Institute AI Engineering", style="bright_black")

    console.print()
    console.print(line0)
    console.print(line1)
    console.print(line2)
    console.print()


def _get_model() -> str:
    try:
        return Configs().default_worker_model  # type: ignore[call-arg]
    except Exception:
        return "gemini-2.5-flash"


def _read_csv_rows(path: Path, row_indices: list[int] | None = None, max_rows: int | None = None) -> list[dict]:
    """Read rows from the transformed data CSV.

    Returns a list of dicts with keys ``title``, ``maintext``, ``mentioned_companies``, ``named_entities``.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row_indices is not None and idx not in row_indices:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            rows.append(row)
    return rows


async def _run_agent_on_article(title: str, maintext: str) -> dict:
    """Run the entity extraction agent on a single article and return parsed output."""
    result = await run_entity_extraction(title, maintext)
    return result.model_dump()


def _display_result(result: dict, title: str, idx: int | None = None, total: int | None = None) -> None:
    """Render extraction results using Rich."""
    header = f"Article: {title[:80]}{'...' if len(title) > 80 else ''}"
    if idx is not None and total is not None:
        header = f"[{idx}/{total}] {header}"

    console.print(Panel(header, border_style="blue", padding=(0, 1)))

    companies = result.get("mentioned_companies", [])
    if companies:
        console.print(f"  [bold]Mentioned Companies:[/bold] [cyan]{', '.join(companies)}[/cyan]")
    else:
        console.print("  [bold]Mentioned Companies:[/bold] [dim]none[/dim]")

    entities = result.get("named_entities", [])
    if entities:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE,
            padding=(0, 1),
        )
        table.add_column("Entity", style="white")
        table.add_column("Group", style="cyan", width=6)
        table.add_column("Normalized", style="yellow")

        for ent in entities:
            word = ent.get("word", "")
            group = ent.get("entity_group", "")
            normalized = ent.get("normalized") or "-"
            table.add_row(word, group, normalized)

        console.print(table)
    else:
        console.print("  [dim]No named entities extracted.[/dim]")

    console.print()


def _display_ground_truth(row: dict) -> None:
    """Show ground-truth columns from the CSV for comparison."""
    gt_companies = row.get("mentioned_companies", "")
    gt_entities = row.get("named_entities", "")

    parts = []
    if gt_companies:
        parts.append(f"[bold]Companies:[/bold] [yellow]{gt_companies}[/yellow]")
    if gt_entities:
        truncated = gt_entities[:200] + "..." if len(gt_entities) > 200 else gt_entities
        parts.append(f"[bold]Entities:[/bold] [dim]{truncated}[/dim]")

    if parts:
        console.print(
            Panel(
                "\n".join(parts),
                title="[bold yellow]Ground Truth[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )


async def cmd_run(
    file: str | None = None,
    row_indices: list[int] | None = None,
    max_rows: int = 1,
    title: str | None = None,
    maintext: str | None = None,
    show_ground_truth: bool = False,
) -> int:
    """Run entity extraction on articles."""
    display_banner()

    if title and maintext:
        console.print("[bold blue]Running on provided article...[/bold blue]\n")
        result = await _run_agent_on_article(title, maintext)
        _display_result(result, title)
        console.print("[bold green]Done[/bold green]")
        return 0

    if not file:
        console.print("[bold red]Error: provide --file or both --title and --maintext[/bold red]")
        return 1

    csv_path = Path(file)
    if not csv_path.exists():
        console.print(f"[bold red]Error: file not found: {csv_path}[/bold red]")
        return 1

    rows = _read_csv_rows(csv_path, row_indices=row_indices, max_rows=max_rows if row_indices is None else None)
    if not rows:
        console.print("[bold red]Error: no matching rows found[/bold red]")
        return 1

    console.print(f"[bold blue]Processing {len(rows)} article(s)...[/bold blue]\n")

    for i, row in enumerate(rows, 1):
        article_title = row.get("title", "")
        article_text = row.get("maintext", "")

        console.print(f"[bold cyan]--- Article {i}/{len(rows)} ---[/bold cyan]")
        with console.status("[bold cyan]Extracting entities...[/bold cyan]"):
            result = await _run_agent_on_article(article_title, article_text)

        _display_result(result, article_title, idx=i, total=len(rows))

        if show_ground_truth:
            _display_ground_truth(row)

    console.print(f"[bold green]Done — processed {len(rows)} article(s)[/bold green]")
    return 0


def _display_help() -> None:
    console.print()
    commands_table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    commands_table.add_column("Command", style="bold green", width=12)
    commands_table.add_column("Description")
    commands_table.add_row("run", "Run entity extraction on articles from CSV or inline text")
    console.print("[bold]Commands:[/bold]")
    console.print(commands_table)
    console.print()
    console.print("[bold]Examples:[/bold]")
    console.print(
        "  [dim]$[/dim] entity-extract [green]run[/green] "
        "[cyan]--file[/cyan] data/transformed_data/2017_data.csv [cyan]--rows[/cyan] 3"
    )
    console.print(
        "  [dim]$[/dim] entity-extract [green]run[/green] "
        "[cyan]--file[/cyan] data/transformed_data/2017_data.csv [cyan]--row-index[/cyan] 0 5 10"
    )
    console.print(
        "  [dim]$[/dim] entity-extract [green]run[/green] "
        "[cyan]--file[/cyan] data/transformed_data/2017_data.csv [cyan]--rows[/cyan] 2 [cyan]--ground-truth[/cyan]"
    )
    console.print(
        '  [dim]$[/dim] entity-extract [green]run[/green] '
        '[cyan]--title[/cyan] [yellow]"Headline"[/yellow] [cyan]--maintext[/cyan] [yellow]"Article body..."[/yellow]'
    )
    console.print()


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(message)s", force=True)
    for name in ["google.adk", "google.genai", "httpx", "httpcore"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(
        prog="entity-extract",
        description="Entity Extraction Agent CLI",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--version", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run entity extraction")
    run_parser.add_argument("--file", type=str, help="Path to transformed data CSV")
    run_parser.add_argument("--rows", type=int, default=1, help="Number of rows to process (default: 1)")
    run_parser.add_argument("--row-index", type=int, nargs="+", metavar="IDX", help="Specific 0-based row indices")
    run_parser.add_argument("--title", type=str, help="Article title (inline mode)")
    run_parser.add_argument("--maintext", type=str, help="Article body text (inline mode)")
    run_parser.add_argument("--ground-truth", action="store_true", help="Show ground-truth from CSV for comparison")

    args = parser.parse_args()

    if args.version:
        display_banner()
        console.print(f"[bold]entity-extract[/bold] v{get_version()}")
        return 0

    if args.command == "run":
        return asyncio.run(
            cmd_run(
                file=args.file,
                row_indices=args.row_index,
                max_rows=args.rows,
                title=args.title,
                maintext=args.maintext,
                show_ground_truth=args.ground_truth,
            )
        )

    display_banner()
    if args.help:
        pass
    _display_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
