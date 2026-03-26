"""Evaluate the Summarization Agent using Langfuse experiments.

This script runs the Summarization Agent against a Langfuse dataset and evaluates
results using LLM-as-a-judge methodology. Results are automatically logged to 
Langfuse for analysis and comparison.

Optionally, trace-level groundedness evaluation can be enabled to check if summaries
are supported by the original article content.

Usage:
    # Run a full evaluation
    python evaluate.py

    # Run with custom dataset and experiment name
    python evaluate.py --dataset-name "MySummarizationDataset" --experiment-name "v2-test"

    # Enable trace groundedness evaluation
    ENABLE_TRACE_GROUNDEDNESS=true python evaluate.py
"""

import asyncio
import logging
import os
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment, run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import create_trace_groundedness_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import EvaluationResult
from aieng.agent_evals.logging_config import setup_logging
from aieng.agent_evals.summarization.agent import SummarizationAgent
from aieng.agent_evals.summarization.summarization_grader import (
    SummarizationResult,
    evaluate_summarization_async,
)
from dotenv import load_dotenv
from langfuse.experiment import Evaluation, ExperimentResult


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "Financial-News-Summarization"
DEFAULT_EXPERIMENT_NAME = "Summarization Agent Evaluation"

# Configuration for trace groundedness evaluation
ENABLE_TRACE_GROUNDEDNESS = os.getenv("ENABLE_TRACE_GROUNDEDNESS", "false").lower() in ("true", "1", "yes")


async def agent_task(*, item: Any, **kwargs: Any) -> str:  # noqa: ARG001
    """Run the Summarization Agent on a dataset item.

    Parameters
    ----------
    item : Any
        The Langfuse experiment item containing the article data.
        Expected format: {"title": str, "maintext": str} or {"input": {"title": str, "maintext": str}}
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    str
        The agent's summary text. Rich execution data (reasoning chain, timing)
        is attached to the Langfuse span metadata.
    """
    # Extract article data from item
    if hasattr(item, 'input') and isinstance(item.input, dict):
        article_data = item.input
    elif isinstance(item, dict):
        article_data = item
    else:
        raise ValueError(f"Unexpected item format: {type(item)}")

    title = article_data.get("title", "")
    maintext = article_data.get("maintext", "")

    if not title and not maintext:
        raise ValueError("Item must contain 'title' and/or 'maintext' fields")

    logger.info(f"Running summarization agent on: {title[:80]}...")

    try:
        agent = SummarizationAgent()
        response = await agent.summarize_async(title=title, body=maintext)
        logger.info(f"Agent completed: {len(response.text)} chars, {response.total_duration_ms}ms")

        # Attach rich execution data to the span metadata
        client_manager = AsyncClientManager.get_instance()
        # Prepare metadata with fallback for empty reasoning chain
        metadata = {
            "total_duration_ms": response.total_duration_ms,
            "article_title": title[:100],  # Truncate long titles
            "article_length": len(maintext),
            "summary_length": len(response.text),
            "has_reasoning_chain": bool(response.reasoning_chain),
        }

        # Add reasoning chain if available, otherwise note it's empty
        if response.reasoning_chain:
            metadata["reasoning_chain"] = response.reasoning_chain[:3]  # Limit to first 3 items
        else:
            metadata["reasoning_note"] = "No explicit reasoning chain generated (normal for simple tasks)"

        client_manager.langfuse_client.update_current_span(metadata=metadata)

        return response.text
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Error: {e}"


async def summarization_evaluator(
    *,
    input: Any,  # noqa: A002
    output: str,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Evaluate the agent's summary using LLM-as-a-judge methodology.

    Parameters
    ----------
    input : Any
        The original article data (title and maintext).
    output : str
        The agent's generated summary.
    expected_output : str
        The ground truth summary (may be used for reference but not strict matching).
    metadata : dict[str, Any] | None, optional
        Item metadata (unused for summarization).
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with accuracy, completeness, conciseness,
        clarity, average score, and overall quality assessments.
    """
    # Extract article data
    if isinstance(input, dict):
        title = input.get("title", "")
        maintext = input.get("maintext", "")
    else:
        # Fallback: try to parse as string or use empty values
        title = ""
        maintext = str(input) if input else ""

    logger.info(f"Evaluating summary for article: {title[:50]}...")

    try:
        # Use the summarization-specific evaluator
        result = await evaluate_summarization_async(
            title=title,
            body=maintext,
            summary=str(output),
            model_config=LLMRequestConfig(temperature=0.0),
        )

        evaluations = result.to_evaluations()
        logger.info(f"Evaluation complete: {result.overall_quality.value} (avg: {result.average_score:.2f})")
        return evaluations

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return SummarizationResult.error_evaluations(str(e))


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int = 1,
    enable_trace_groundedness: bool = False,
) -> ExperimentResult | EvaluationResult:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    enable_trace_groundedness : bool, optional
        Whether to enable trace-level groundedness evaluation, by default False.
    """
    client_manager = AsyncClientManager.get_instance()

    try:
        logger.info(f"Starting experiment '{experiment_name}' on dataset '{dataset_name}'")
        logger.info(f"Max concurrency: {max_concurrency}")
        logger.info(f"Trace groundedness: {'enabled' if enable_trace_groundedness else 'disabled'}")

        result: ExperimentResult | EvaluationResult
        if enable_trace_groundedness:
            # Create trace groundedness evaluator
            groundedness_evaluator = create_trace_groundedness_evaluator(
                name="trace_groundedness",
                model_config=LLMRequestConfig(temperature=0.0),
            )

            # Run with trace evaluations
            result = run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Summarization Agent evaluation with LLM-as-a-judge and trace groundedness",
                task=agent_task,
                evaluators=[summarization_evaluator],  # Item-level evaluators
                trace_evaluators=[groundedness_evaluator],  # Trace-level evaluators
                max_concurrency=max_concurrency,
            )
        else:
            # Run without trace evaluations
            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Summarization Agent evaluation with LLM-as-a-judge",
                task=agent_task,
                evaluators=[summarization_evaluator],
                max_concurrency=max_concurrency,
            )

        logger.info("Experiment complete!")
        # Handle both ExperimentResult and EvaluationResult
        if isinstance(result, EvaluationResult):
            # EvaluationResult from run_experiment_with_trace_evals
            logger.info(f"Results: {result.experiment}")
            if result.trace_evaluations:
                trace_evals = result.trace_evaluations
                logger.info(
                    f"Trace evaluations: {len(trace_evals.evaluations_by_trace_id)} traces, "
                    f"{len(trace_evals.skipped_trace_ids)} skipped, {len(trace_evals.failed_trace_ids)} failed"
                )
        else:
            # ExperimentResult from run_experiment
            logger.info(f"Results: {result}")

        return result

    finally:
        logger.info("Closing client manager and flushing data...")
        try:
            await client_manager.close()
            await asyncio.sleep(0.1)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--experiment-name",
    default=DEFAULT_EXPERIMENT_NAME,
    help="Name for this experiment run.",
)
@click.option(
    "--max-concurrency",
    default=1,
    type=int,
    help="Maximum concurrent agent runs (default: 1).",
)
@click.option(
    "--enable-trace-groundedness",
    is_flag=True,
    default=ENABLE_TRACE_GROUNDEDNESS,
    help="Enable trace-level groundedness evaluation.",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int, enable_trace_groundedness: bool) -> None:
    """Run Summarization Agent evaluation using Langfuse experiments."""
    asyncio.run(
        run_evaluation(
            dataset_name,
            experiment_name,
            max_concurrency,
            enable_trace_groundedness,
        )
    )


if __name__ == "__main__":
    cli()