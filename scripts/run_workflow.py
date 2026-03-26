"""Workflow entry point for the Knowledge QA evaluation pipeline.

Loads financial news articles from a local CSV **or** a Langfuse dataset, runs
entity extraction and/or summarization agents (parallel by default; use
``--sequential`` for one-at-a-time), evaluates results automatically, and
pushes evaluation scores to Langfuse when ``--traces`` is enabled.

Usage
-----
    python scripts/run_workflow.py
    python scripts/run_workflow.py --data-file data/transformed_data/2018_data.csv --sample-size 10
    python scripts/run_workflow.py --dataset-name FinancialNews-2017 --sample-size 5
    python scripts/run_workflow.py --agents entity-extraction --sequential
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.evaluation.trace import flush_traces
from aieng.agent_evals.summarization import SummarizationAgent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

_EVAL_RUNNER_DIR = str(Path(__file__).resolve().parent.parent / "implementations" / "summarization")
if _EVAL_RUNNER_DIR not in sys.path:
    sys.path.insert(0, _EVAL_RUNNER_DIR)

from eval_runner import run_entity_extraction_evals, run_summarization_evals  # noqa: E402

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
DEFAULT_DATASET_NAME = "FinancialNews-2017"
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
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    context_used_percent: float = 0.0


class SummarizationResult(BaseModel):
    """Result from the summarization agent for a single article."""

    article_index: int
    summary: str = ""
    duration_ms: int = 0
    error: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    context_used_percent: float = 0.0


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


def load_data_from_langfuse(
    dataset_name: str,
    sample_size: int | None = None,
) -> list[ArticleRecord]:
    """Fetch articles from a Langfuse dataset.

    Each dataset item is expected to have ``input.title``, ``input.maintext``,
    and optionally ``input.description``.  Entity-extraction ground truth is
    read from ``metadata.ground_truth_format`` (a JSON string with
    ``mentioned_companies`` and ``named_entities`` keys).

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset.
    sample_size : int, optional
        If provided, only return the first *sample_size* records.
    """
    from aieng.agent_evals.async_client_manager import AsyncClientManager

    manager = AsyncClientManager.get_instance()
    lf = manager.langfuse_client
    dataset = lf.get_dataset(dataset_name)

    records: list[ArticleRecord] = []
    for item in dataset.items:
        input_data = item.input
        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        title = (input_data.get("title") or "") if isinstance(input_data, dict) else ""
        maintext = (input_data.get("maintext") or "") if isinstance(input_data, dict) else ""
        description = (input_data.get("description") or "") if isinstance(input_data, dict) else ""

        mentioned_companies: list[str] = []
        named_entities: list[dict[str, Any]] = []

        metadata = item.metadata or {}
        gt_raw = metadata.get("ground_truth_format")
        if gt_raw:
            gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
            mentioned_companies = gt.get("mentioned_companies", [])
            named_entities = gt.get("named_entities", [])

        if not maintext.strip():
            continue

        records.append(
            ArticleRecord(
                title=str(title),
                maintext=str(maintext),
                description=str(description),
                mentioned_companies=mentioned_companies if isinstance(mentioned_companies, list) else [],
                named_entities=named_entities if isinstance(named_entities, list) else [],
            )
        )

        if sample_size is not None and len(records) >= sample_size:
            break

    return records


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------


async def run_entity_extraction(
    articles: list[ArticleRecord], *, langfuse_tracing: bool = False,
) -> list[EntityExtractionResult]:
    """Run the entity extraction agent over all articles sequentially.

    Delegates to ``agent.py``'s ``run_entity_extraction`` which handles
    session management, robust JSON parsing, and token tracking internally.
    Retries once on failure before recording an error.
    """
    from aieng.agent_evals.entity_extraction.agent import (
        run_entity_extraction as _extract,
    )

    max_retries = 2
    results: list[EntityExtractionResult] = []
    for i, article in enumerate(articles):
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await _extract(
                    article.title,
                    article.maintext,
                    langfuse_tracing=langfuse_tracing,
                )
                u = response.token_usage
                token_fields: dict[str, Any] = {}
                if u:
                    token_fields = {
                        "total_prompt_tokens": u.total_prompt_tokens,
                        "total_completion_tokens": u.total_completion_tokens,
                        "total_tokens": u.total_tokens,
                        "context_used_percent": u.context_used_percent,
                    }
                results.append(
                    EntityExtractionResult(
                        article_index=i,
                        mentioned_companies=response.output.mentioned_companies,
                        named_entities=[
                            e.model_dump()
                            for e in response.output.named_entities
                        ],
                        duration_ms=response.total_duration_ms,
                        **token_fields,
                    )
                )
                token_info = (
                    f", tokens: {u.total_tokens}"
                    if u and u.total_tokens else ""
                )
                console.print(
                    f"  [green]Entity extraction[/green] article "
                    f"{i + 1}/{len(articles)} "
                    f"({response.total_duration_ms}ms{token_info})"
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "Article %d: attempt %d failed (%s), retrying…",
                        i, attempt, exc,
                    )

        if last_err is not None:
            logger.error(
                "Entity extraction failed for article %d: %s", i, last_err,
            )
            results.append(
                EntityExtractionResult(
                    article_index=i, error=str(last_err),
                )
            )
            console.print(
                f"  [red]Entity extraction[/red] article "
                f"{i + 1}/{len(articles)} FAILED: {last_err}"
            )

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
                token_fields: dict[str, Any] = {}
                if response.token_usage:
                    u = response.token_usage
                    token_fields = {
                        "total_prompt_tokens": u.total_prompt_tokens,
                        "total_completion_tokens": u.total_completion_tokens,
                        "total_tokens": u.total_tokens,
                        "context_used_percent": u.context_used_percent,
                    }
                results.append(
                    SummarizationResult(
                        article_index=i,
                        summary=response.text,
                        duration_ms=duration_ms,
                        **token_fields,
                    )
                )
                token_info = (
                    f", tokens: {token_fields.get('total_tokens', 0)}"
                    if token_fields
                    else ""
                )
                console.print(f"  [blue]Summarization[/blue] article {i + 1}/{len(articles)} ({duration_ms}ms{token_info})")
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
# Pipeline orchestration (with immediate evaluation)
# ---------------------------------------------------------------------------


async def _ee_branch(
    articles: list[ArticleRecord],
    *,
    langfuse_tracing: bool = False,
    run_context: dict[str, Any] | None = None,
) -> list[EntityExtractionResult]:
    """Run entity extraction agent then evaluate immediately."""
    results = await run_entity_extraction(articles, langfuse_tracing=langfuse_tracing)
    if results:
        run_entity_extraction_evals(
            results, articles, langfuse=langfuse_tracing, run_context=run_context,
        )
    return results


async def _sum_branch(
    articles: list[ArticleRecord],
    *,
    langfuse_tracing: bool = False,
    run_context: dict[str, Any] | None = None,
) -> list[SummarizationResult]:
    """Run summarization agent then evaluate immediately."""
    results = await run_summarization(articles, langfuse_tracing=langfuse_tracing)
    if results:
        await run_summarization_evals(
            results, articles, langfuse=langfuse_tracing, run_context=run_context,
        )
    return results


async def run_pipeline(
    articles: list[ArticleRecord],
    agents_to_run: list[str],
    *,
    sequential: bool = False,
    langfuse_tracing: bool = False,
    run_context: dict[str, Any] | None = None,
) -> tuple[list[EntityExtractionResult], list[SummarizationResult]]:
    """Run selected agent pipelines with immediate evaluation.

    By default both branches run concurrently.  Pass ``sequential=True``
    to run one after the other (useful when rate-limited).
    """
    entity_results: list[EntityExtractionResult] = []
    summarization_results: list[SummarizationResult] = []

    branch_kw: dict[str, Any] = {
        "langfuse_tracing": langfuse_tracing,
        "run_context": run_context,
    }

    if sequential:
        if "entity-extraction" in agents_to_run:
            entity_results = await _ee_branch(articles, **branch_kw)
        if "summarization" in agents_to_run:
            summarization_results = await _sum_branch(articles, **branch_kw)
        return entity_results, summarization_results

    tasks: dict[str, asyncio.Task[Any]] = {}
    if "entity-extraction" in agents_to_run:
        tasks["entity-extraction"] = asyncio.create_task(
            _ee_branch(articles, **branch_kw)
        )
    if "summarization" in agents_to_run:
        tasks["summarization"] = asyncio.create_task(
            _sum_branch(articles, **branch_kw)
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

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-file",
        default=None,
        help=f"Path to input CSV file (default: {DEFAULT_DATA_FILE})",
    )
    data_group.add_argument(
        "--dataset-name",
        default=None,
        help=(
            f"Name of a Langfuse dataset to load articles from "
            f"(default when used: {DEFAULT_DATASET_NAME})"
        ),
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
        "--sequential",
        action="store_true",
        help="Run agent pipelines one after the other instead of concurrently",
    )
    parser.add_argument(
        "--traces",
        action="store_true",
        help="Enable Langfuse tracing via OpenTelemetry and push evaluation scores to Langfuse",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point that orchestrates the full workflow."""
    start_time = time.time()

    langfuse_tracing = getattr(args, "traces", False)

    # --- Data loading (CSV or Langfuse dataset) ---
    if args.dataset_name is not None:
        dataset_name = args.dataset_name or DEFAULT_DATASET_NAME
        console.print(f"\n[bold]Loading data from Langfuse dataset[/bold] {dataset_name}")
        articles = load_data_from_langfuse(dataset_name, args.sample_size)
        data_source_label = f"langfuse:{dataset_name}"
    else:
        data_file = args.data_file or DEFAULT_DATA_FILE
        console.print(f"\n[bold]Loading data from[/bold] {data_file}")
        articles = load_data(data_file, args.sample_size)
        data_source_label = data_file

    console.print(f"  Loaded {len(articles)} articles")

    # --- Workflow run identity (shared across all Langfuse eval traces) ---
    workflow_run_id = f"workflow-run-{uuid.uuid4().hex[:12]}"
    run_context: dict[str, Any] = {
        "source": "workflow",
        "run_id": workflow_run_id,
        "data_source": data_source_label,
        "agents": args.agents,
        "mode": "sequential" if args.sequential else "parallel",
    }
    console.print(f"  Run ID: {workflow_run_id}")

    # --- Run agent pipelines (evaluation happens inside each branch) ---
    mode = run_context["mode"]
    console.print(f"\n[bold]Running agents ({mode}):[/bold] {', '.join(args.agents)}")
    entity_results, summarization_results = await run_pipeline(
        articles,
        args.agents,
        sequential=args.sequential,
        langfuse_tracing=langfuse_tracing,
        run_context=run_context,
    )

    if langfuse_tracing:
        flush_traces()

    total_duration_ms = int((time.time() - start_time) * 1000)

    workflow_result = WorkflowResult(
        data_file=data_source_label,
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
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
