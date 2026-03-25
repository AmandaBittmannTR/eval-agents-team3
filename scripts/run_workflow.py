"""Workflow entry point for the Knowledge QA evaluation pipeline.

Loads financial news articles from a CSV file, runs entity extraction and
summarization agents (sequential by default; use ``--parallel`` for concurrent
runs), collects structured results, and provides hooks for downstream evaluation.

Usage
-----
    python scripts/run_workflow.py
    python scripts/run_workflow.py --data-file data/transformed_data/2018_data.csv --sample-size 10
    python scripts/run_workflow.py --agents entity-extraction
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.entity_extraction import EntityExtractionOutput, create_entity_extraction_agent
from aieng.agent_evals.evaluation.trace import flush_traces
from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.summarization import SummarizationAgent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


def _env_key_nonempty(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def ensure_google_genai_env() -> None:
    """Mirror API keys from ``Configs`` into ``os.environ`` for ADK / google-genai.

    ``google.genai`` uses ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` from
    ``os.environ`` (see ``get_env_api_key()`` in the SDK). Values must be
    non-empty: ``""`` counts as missing.

    ``pydantic-settings`` can load a key from ``OPENAI_API_KEY`` into ``Configs``
    while leaving ``GOOGLE_API_KEY`` unset. Also, ``.env`` lines like
    ``GOOGLE_API_KEY=`` set an empty string; ``setdefault`` does not override those,
    which produced "Missing key inputs argument" even when another var had the key.
    """
    cfg = Configs()  # type: ignore[call-arg]
    key = cfg.openai_api_key.get_secret_value().strip()
    if not key:
        return
    if not _env_key_nonempty("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key
    if not _env_key_nonempty("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = key


DEFAULT_DATA_FILE = "data/transformed_data/2017_data.csv"
DEFAULT_OUTPUT_DIR = "outputs"
REQUIRED_COLUMNS = {"title", "maintext", "description", "mentioned_companies", "named_entities"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ArticleRecord(BaseModel):
    """A single news article with gold-standard labels for evaluation.

    Fields
    ------
    title, maintext : str
        Agent input (Blue in the architecture diagram).
    description : str
        Gold standard for summarization evaluation (Yellow).
    mentioned_companies, named_entities : list
        Gold standard for entity extraction evaluation (Orange).
    """

    title: str = ""
    maintext: str
    description: str = ""
    mentioned_companies: list[str] = Field(default_factory=list)
    named_entities: list[dict[str, Any]] = Field(default_factory=list)


class EntityExtractionResult(BaseModel):
    """Result from the entity extraction agent for a single article."""

    article_index: int
    mentioned_companies: list[str] = Field(default_factory=list)
    named_entities: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None


class SummarizationResult(BaseModel):
    """Result from the summarization agent for a single article."""

    article_index: int
    summary: str = ""
    duration_ms: int = 0
    error: str | None = None


class WorkflowResult(BaseModel):
    """Aggregated results from the full workflow run."""

    data_file: str
    total_articles: int
    entity_extraction_results: list[EntityExtractionResult] = Field(default_factory=list)
    summarization_results: list[SummarizationResult] = Field(default_factory=list)
    total_duration_ms: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _safe_literal_eval(value: Any) -> Any:
    """Parse a stringified Python literal, returning an empty list on failure."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def load_data(path: str, sample_size: int | None = None) -> list[ArticleRecord]:
    """Load articles from a transformed-data CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    sample_size : int, optional
        If provided, only return the first *sample_size* records.

    Returns
    -------
    list[ArticleRecord]
        Validated article records ready for agent processing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if sample_size is not None:
        df = df.head(sample_size)

    df["mentioned_companies"] = df["mentioned_companies"].apply(_safe_literal_eval)
    df["named_entities"] = df["named_entities"].apply(_safe_literal_eval)
    df["title"] = df["title"].fillna("")
    df["maintext"] = df["maintext"].fillna("")
    df["description"] = df["description"].fillna("")

    records: list[ArticleRecord] = []
    for _, row in df.iterrows():
        records.append(
            ArticleRecord(
                title=str(row["title"]),
                maintext=str(row["maintext"]),
                description=str(row["description"]),
                mentioned_companies=row["mentioned_companies"] if isinstance(row["mentioned_companies"], list) else [],
                named_entities=row["named_entities"] if isinstance(row["named_entities"], list) else [],
            )
        )
    return records


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------


async def run_entity_extraction(
    articles: list[ArticleRecord], *, langfuse_tracing: bool = False
) -> list[EntityExtractionResult]:
    """Run the entity extraction agent over all articles sequentially.

    Each article gets a fresh ADK session. The agent's ``output_schema`` is
    ``EntityExtractionOutput``, so the final response is structured JSON that
    we parse back into fields for the result model.
    """
    if langfuse_tracing:
        init_tracing(service_name="EntityExtractionAgent")

    agent = create_entity_extraction_agent()
    session_service = InMemorySessionService()
    runner = Runner(app_name="entity_extraction", agent=agent, session_service=session_service)

    results: list[EntityExtractionResult] = []
    try:
        for i, article in enumerate(articles):
            start = time.time()
            try:
                session = await session_service.create_session(
                    app_name="entity_extraction", user_id="workflow", state={}
                )
                prompt = json.dumps({"title": article.title, "maintext": article.maintext})
                content = types.Content(role="user", parts=[types.Part(text=prompt)])

                final_text = ""
                async for event in runner.run_async(
                    user_id="workflow", session_id=session.id, new_message=content
                ):
                    if hasattr(event, "is_final_response") and event.is_final_response():
                        if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
                            for part in event.content.parts:
                                if not getattr(part, "thought", False) and hasattr(part, "text") and part.text:
                                    final_text = part.text

                duration_ms = int((time.time() - start) * 1000)

                output: EntityExtractionOutput | None = None
                if final_text.strip():
                    try:
                        output = EntityExtractionOutput.model_validate_json(final_text.strip())
                    except Exception:
                        output = EntityExtractionOutput.model_validate(json.loads(final_text.strip()))

                results.append(
                    EntityExtractionResult(
                        article_index=i,
                        mentioned_companies=output.mentioned_companies if output else [],
                        named_entities=[e.model_dump() for e in output.named_entities] if output else [],
                        duration_ms=duration_ms,
                    )
                )
                console.print(
                    f"  [green]Entity extraction[/green] article {i + 1}/{len(articles)} ({duration_ms}ms)"
                )
            except Exception as exc:
                duration_ms = int((time.time() - start) * 1000)
                logger.error(f"Entity extraction failed for article {i}: {exc}")
                results.append(
                    EntityExtractionResult(article_index=i, duration_ms=duration_ms, error=str(exc))
                )
                console.print(f"  [red]Entity extraction[/red] article {i + 1}/{len(articles)} FAILED: {exc}")
    finally:
        await runner.close()

    return results


async def run_summarization(
    articles: list[ArticleRecord], *, langfuse_tracing: bool = False
) -> list[SummarizationResult]:
    """Run the summarization agent over all articles sequentially.

    Uses the ``SummarizationAgent`` which accepts ``(title, body)`` and
    returns a ``SummarizationResponse`` with ``.text`` and ``.total_duration_ms``.
    """
    agent = SummarizationAgent(langfuse_tracing=langfuse_tracing)
    results: list[SummarizationResult] = []
    try:
        for i, article in enumerate(articles):
            start = time.time()
            try:
                # Fresh ADK session per article. Do not call agent.reset() here: it recreates Runner
                # without closing the old one, which leaks clients and triggers genai aclose() errors.
                response = await agent.summarize_async(
                    title=article.title,
                    body=article.maintext,
                    session_id=f"workflow-{i}-{uuid.uuid4().hex[:8]}",
                )
                duration_ms = response.total_duration_ms or int((time.time() - start) * 1000)
                results.append(
                    SummarizationResult(article_index=i, summary=response.text, duration_ms=duration_ms)
                )
                console.print(f"  [blue]Summarization[/blue] article {i + 1}/{len(articles)} ({duration_ms}ms)")
            except Exception as exc:
                duration_ms = int((time.time() - start) * 1000)
                logger.error(f"Summarization failed for article {i}: {exc}")
                results.append(
                    SummarizationResult(article_index=i, duration_ms=duration_ms, error=str(exc))
                )
                console.print(f"  [red]Summarization[/red] article {i + 1}/{len(articles)} FAILED: {exc}")
    finally:
        await agent.aclose()

    return results


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


async def run_pipeline(
    articles: list[ArticleRecord],
    agents_to_run: list[str],
    *,
    parallel: bool = False,
    langfuse_tracing: bool = False,
) -> tuple[list[EntityExtractionResult], list[SummarizationResult]]:
    """Run selected agent pipelines (sequential by default, optional parallel)."""
    entity_results: list[EntityExtractionResult] = []
    summarization_results: list[SummarizationResult] = []

    if not parallel:
        if "entity-extraction" in agents_to_run:
            entity_results = await run_entity_extraction(articles, langfuse_tracing=langfuse_tracing)
        if "summarization" in agents_to_run:
            summarization_results = await run_summarization(articles, langfuse_tracing=langfuse_tracing)
        return entity_results, summarization_results

    tasks: dict[str, asyncio.Task[Any]] = {}
    if "entity-extraction" in agents_to_run:
        tasks["entity-extraction"] = asyncio.create_task(
            run_entity_extraction(articles, langfuse_tracing=langfuse_tracing)
        )
    if "summarization" in agents_to_run:
        tasks["summarization"] = asyncio.create_task(
            run_summarization(articles, langfuse_tracing=langfuse_tracing)
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for key, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            console.print(f"[red]Pipeline '{key}' failed: {result}[/red]")
            continue
        if key == "entity-extraction":
            entity_results = result
        elif key == "summarization":
            summarization_results = result

    return entity_results, summarization_results


# ---------------------------------------------------------------------------
# Evaluation placeholders
# ---------------------------------------------------------------------------


def evaluate_entity_extraction(
    results: list[EntityExtractionResult],
    articles: list[ArticleRecord],
) -> None:
    """Placeholder for code-based entity extraction evaluation.

    Will compare ``results[i].mentioned_companies`` / ``results[i].named_entities``
    against ``articles[i].mentioned_companies`` / ``articles[i].named_entities``
    using precision, recall, and F1 metrics.
    """
    successful = sum(1 for r in results if r.error is None)
    console.print("\n[bold]Entity Extraction Evaluation[/bold] (placeholder)")
    console.print(f"  {successful}/{len(results)} articles processed successfully")
    console.print("  TODO: implement code-based metrics (precision, recall, F1)")


def evaluate_summarization(
    results: list[SummarizationResult],
    articles: list[ArticleRecord],
) -> None:
    """Placeholder for LLM-as-a-Judge summarization evaluation.

    Will compare ``results[i].summary`` against ``articles[i].description``
    using condenseness and completeness rubrics via a Gemini evaluator model.
    """
    successful = sum(1 for r in results if r.error is None)
    console.print("\n[bold]Summarization Evaluation[/bold] (placeholder)")
    console.print(f"  {successful}/{len(results)} articles processed successfully")
    console.print("  TODO: implement LLM-as-a-Judge evaluation (condenseness, completeness)")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(result: WorkflowResult, output_dir: str) -> Path:
    """Write workflow results to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "workflow_result.json"
    output_file.write_text(result.model_dump_json(indent=2))
    return output_file


def print_summary(result: WorkflowResult) -> None:
    """Print a rich summary table to the console."""
    table = Table(title="Workflow Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Data file", result.data_file)
    table.add_row("Articles processed", str(result.total_articles))

    if result.entity_extraction_results:
        ok = sum(1 for r in result.entity_extraction_results if r.error is None)
        table.add_row("Entity extraction", f"{ok}/{len(result.entity_extraction_results)}")

    if result.summarization_results:
        ok = sum(1 for r in result.summarization_results if r.error is None)
        table.add_row("Summarization", f"{ok}/{len(result.summarization_results)}")

    total_sec = result.total_duration_ms / 1000
    table.add_row("Total duration", f"{total_sec:.1f}s")

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the workflow CLI."""
    parser = argparse.ArgumentParser(
        description="Run the Knowledge QA evaluation workflow (entity extraction + summarization).",
    )
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help=f"Path to input CSV file (default: {DEFAULT_DATA_FILE})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit the number of articles to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write results JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["entity-extraction", "summarization"],
        default=["entity-extraction", "summarization"],
        help="Which agent pipelines to run (default: both)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run entity extraction and summarization concurrently (default: run one after the other)",
    )
    parser.add_argument(
        "--langfuse-trace",
        action="store_true",
        help="Enable Langfuse tracing via OpenTelemetry for both agents",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point that orchestrates the full workflow."""
    start_time = time.time()

    console.print(f"\n[bold]Loading data from[/bold] {args.data_file}")
    articles = load_data(args.data_file, args.sample_size)
    console.print(f"  Loaded {len(articles)} articles")

    langfuse_tracing = getattr(args, "langfuse_trace", False)

    console.print(f"\n[bold]Running agents:[/bold] {', '.join(args.agents)}")
    entity_results, summarization_results = await run_pipeline(
        articles, args.agents, parallel=args.parallel, langfuse_tracing=langfuse_tracing
    )

    if langfuse_tracing:
        flush_traces()

    if entity_results:
        evaluate_entity_extraction(entity_results, articles)
    if summarization_results:
        evaluate_summarization(summarization_results, articles)

    total_duration_ms = int((time.time() - start_time) * 1000)

    workflow_result = WorkflowResult(
        data_file=args.data_file,
        total_articles=len(articles),
        entity_extraction_results=entity_results,
        summarization_results=summarization_results,
        total_duration_ms=total_duration_ms,
    )

    output_file = write_results(workflow_result, args.output_dir)
    console.print(f"\n[bold]Results written to[/bold] {output_file}")

    print_summary(workflow_result)


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    ensure_google_genai_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
