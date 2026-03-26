"""Unified evaluation runner for the workflow pipeline.

Bridges workflow data models (``ArticleRecord``, ``EntityExtractionResult``,
``SummarizationResult``) from ``scripts/run_workflow.py`` to the standalone
evaluation scripts and optionally pushes scores to Langfuse.

Three evaluation strategies are provided:

1. **Entity extraction** -- set-based P/R/F1 for companies, entities, and
   words (delegates to ``scripts/entity_extraction_eval.py``).
2. **Summarization similarity** -- TF-IDF cosine + BERTScore F1 (delegates
   to ``scripts/summary_similarity_eval.py``).
3. **Summarization LLM-as-judge** -- accuracy, completeness, conciseness,
   and clarity via ``evaluate_summarization_async`` from the grader package.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import entity_extraction_eval as _ee_eval  # noqa: E402
import summary_similarity_eval as _sim_eval  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()


def _clip_text(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return f"{t[: max_chars - 3]}..."


# ---------------------------------------------------------------------------
# Type bridging helpers
# ---------------------------------------------------------------------------


def _build_ee_ground_and_agent(
    results: list[Any],
    articles: list[Any],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Convert workflow types to entity-extraction eval's keyed-dict format.

    Ground-truth list fields are serialised to JSON strings so that the
    existing ``parse_mentioned_companies`` / ``parse_named_entities`` helpers
    can parse them identically to raw CSV values.
    """
    ground: dict[str, dict[str, str]] = {}
    agent: dict[str, dict[str, Any]] = {}

    for result in results:
        if result.error is not None:
            continue
        idx = result.article_index
        article = articles[idx]
        aid = f"article:{idx}"

        gt_companies = article.mentioned_companies
        gt_entities = article.named_entities
        ag_companies = result.mentioned_companies
        ag_entities = result.named_entities

        if idx == 0:
            logger.debug(
                "=== DIAGNOSTIC (article 0) ===\n"
                "  GT companies (%d): %s\n"
                "  AG companies (%d): %s\n"
                "  GT entities  (%d): %s\n"
                "  AG entities  (%d): %s",
                len(gt_companies), gt_companies,
                len(ag_companies), ag_companies,
                len(gt_entities),
                [e.get("word") for e in gt_entities] if gt_entities else [],
                len(ag_entities),
                [
                    e.get("word") if isinstance(e, dict) else e
                    for e in ag_entities
                ] if ag_entities else [],
            )

        ground[aid] = {
            "mentioned_companies": json.dumps(gt_companies),
            "named_entities": json.dumps(gt_entities),
            "title": article.title,
            "maintext": article.maintext,
        }
        agent[aid] = {
            "mentioned_companies": ag_companies,
            "named_entities": ag_entities,
        }

    return ground, agent


def _build_sim_ground_and_agent(
    results: list[Any],
    articles: list[Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Convert workflow types to similarity eval's dict format."""
    ground: dict[str, dict[str, str]] = {}
    agent: dict[str, str] = {}

    for result in results:
        if result.error is not None or not result.summary.strip():
            continue
        idx = result.article_index
        article = articles[idx]
        aid = f"article:{idx}"

        ground[aid] = {
            "description": article.description,
            "maintext": article.maintext,
            "title": article.title,
        }
        agent[aid] = result.summary

    return ground, agent


# ---------------------------------------------------------------------------
# Entity extraction evaluation
# ---------------------------------------------------------------------------


def evaluate_entity_extraction(
    results: list[Any],
    articles: list[Any],
    *,
    langfuse: bool = True,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run P/R/F1 scoring on entity extraction results.

    Returns
    -------
    dict
        ``{"per_item": [...], "aggregates": {...}}``
    """
    ground, agent = _build_ee_ground_and_agent(results, articles)
    if not agent:
        logger.warning("No successful entity extraction results to evaluate.")
        return {"per_item": [], "aggregates": {}}

    per_item, aggregates = _ee_eval.run_eval(ground, agent)

    if not per_item:
        logger.warning("No matched rows between EE outputs and ground truth.")
        return {"per_item": [], "aggregates": {}}

    console.print("\n[bold]Entity Extraction Evaluation[/bold]")
    for row in per_item:
        aid = row["article_id"]
        console.print(
            f"  {aid}\t"
            f"co_f1={row['companies_f1']:.4f}\t"
            f"ent_f1={row['entities_f1']:.4f}\t"
            f"word_f1={row['word_f1']:.4f}"
        )
    console.print(f"  [dim]{json.dumps(aggregates, indent=2)}[/dim]")

    if langfuse:
        run_metadata: dict[str, Any] = {
            "n_articles": len(results),
            **(run_context or {"source": "workflow"}),
        }
        _ee_eval.push_entity_extraction_eval_to_langfuse(
            rows=per_item,
            aggregates=aggregates,
            ground=ground,
            agent_outputs=agent,
            run_metadata=run_metadata,
        )

    return {"per_item": per_item, "aggregates": aggregates}


# ---------------------------------------------------------------------------
# Summarization similarity evaluation
# ---------------------------------------------------------------------------


def evaluate_summarization_similarity(
    results: list[Any],
    articles: list[Any],
    *,
    langfuse: bool = True,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run TF-IDF cosine similarity + BERTScore evaluation on summaries.

    Returns
    -------
    dict
        ``{"per_item": [...], "aggregates": {...}}``
    """
    ground, agent = _build_sim_ground_and_agent(results, articles)
    if not agent:
        logger.warning(
            "No successful summarization results for similarity eval.",
        )
        return {"per_item": [], "aggregates": {}}

    rows, aggregates = _sim_eval.run_eval(
        ground, agent, bert_model=None, device=None, batch_size=None,
    )

    if not rows:
        logger.warning("No matched rows between summaries and ground truth.")
        return {"per_item": [], "aggregates": {}}

    console.print("\n[bold]Summarization Similarity Evaluation[/bold]")
    for row in rows:
        aid = row["article_id"]
        console.print(
            f"  {aid}\t"
            f"cosine={row['cosine_tfidf_vs_reference_summary']:.4f}\t"
            f"bert_f1={row['bertscore_f1_vs_article']:.4f}"
        )
    console.print(f"  [dim]{json.dumps(aggregates, indent=2)}[/dim]")

    if langfuse:
        run_metadata: dict[str, Any] = {
            "n_articles": len(results),
            **(run_context or {"source": "workflow"}),
        }
        _sim_eval.push_summary_similarity_eval_to_langfuse(
            rows=rows,
            aggregates=aggregates,
            ground=ground,
            agent_summaries=agent,
            run_metadata=run_metadata,
        )

    return {"per_item": rows, "aggregates": aggregates}


# ---------------------------------------------------------------------------
# Summarization LLM-as-judge evaluation
# ---------------------------------------------------------------------------


async def evaluate_summarization_llm_judge(
    results: list[Any],
    articles: list[Any],
    *,
    langfuse: bool = True,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run LLM-as-a-judge grading on summarization results.

    Uses ``evaluate_summarization_async`` from the summarization grader to
    score each summary on accuracy, completeness, conciseness, and clarity.

    Returns
    -------
    dict
        ``{"per_article": [...], "aggregates": {...}}``
    """
    from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
    from aieng.agent_evals.summarization.summarization_grader import (
        evaluate_summarization_async,
    )

    per_article: list[dict[str, Any]] = []

    for result in results:
        if result.error is not None or not result.summary.strip():
            continue
        article = articles[result.article_index]
        aid = f"article:{result.article_index}"

        logger.info("LLM-as-judge evaluating %s ...", aid)
        try:
            grader_result = await evaluate_summarization_async(
                title=article.title,
                body=article.maintext,
                summary=result.summary,
                model_config=LLMRequestConfig(temperature=0.0),
            )
            per_article.append({
                "article_id": aid,
                "article_index": result.article_index,
                "accuracy": grader_result.accuracy,
                "completeness": grader_result.completeness,
                "conciseness": grader_result.conciseness,
                "clarity": grader_result.clarity,
                "average_score": grader_result.average_score,
                "overall_quality": grader_result.overall_quality.value,
                "explanation": grader_result.explanation,
            })
        except Exception:
            logger.exception("LLM-as-judge failed for %s", aid)

    if not per_article:
        logger.warning("No summaries were evaluated by LLM-as-judge.")
        return {"per_article": [], "aggregates": {}}

    n = len(per_article)
    aggregates: dict[str, float] = {
        "mean_accuracy": sum(r["accuracy"] for r in per_article) / n,
        "mean_completeness": sum(r["completeness"] for r in per_article) / n,
        "mean_conciseness": sum(r["conciseness"] for r in per_article) / n,
        "mean_clarity": sum(r["clarity"] for r in per_article) / n,
        "mean_average_score": sum(r["average_score"] for r in per_article) / n,
        "n_evaluated": float(n),
    }

    console.print("\n[bold]Summarization LLM-as-Judge Evaluation[/bold]")
    for row in per_article:
        console.print(
            f"  {row['article_id']}\t"
            f"avg={row['average_score']:.2f}\t"
            f"quality={row['overall_quality']}"
        )
    console.print(f"  [dim]{json.dumps(aggregates, indent=2)}[/dim]")

    if langfuse:
        run_metadata: dict[str, Any] = {
            "n_articles": len(results),
            **(run_context or {"source": "workflow"}),
        }
        _push_llm_judge_eval_to_langfuse(
            rows=per_article,
            aggregates=aggregates,
            articles=articles,
            results=results,
            run_metadata=run_metadata,
        )

    return {"per_article": per_article, "aggregates": aggregates}


def _push_llm_judge_eval_to_langfuse(
    *,
    rows: list[dict[str, Any]],
    aggregates: dict[str, float],
    articles: list[Any],
    results: list[Any],
    run_metadata: dict[str, Any],
) -> None:
    """Push LLM-as-judge scores to Langfuse as a trace with per-article spans."""
    from aieng.agent_evals.async_client_manager import AsyncClientManager

    manager = AsyncClientManager.get_instance()
    cfg = manager.configs
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        logger.warning("Langfuse keys not set; skipping LLM-judge upload.")
        return

    lf = manager.langfuse_client
    try:
        if not lf.auth_check():
            logger.warning("Langfuse auth failed; skipping LLM-judge upload.")
            return
    except Exception:
        logger.exception("Langfuse auth_check failed; skipping upload.")
        return

    session_id = run_metadata.get(
        "run_id", f"llm_judge_eval-{uuid.uuid4().hex[:12]}"
    )
    tags = ["llm_judge_eval", "summarization", "bootcamp"]
    if run_metadata.get("source") == "workflow":
        tags.append("Full Workflow Pipeline")
    trace_id_for_url: str | None = None

    summary_by_index: dict[int, str] = {}
    for r in results:
        if r.error is None:
            summary_by_index[r.article_index] = r.summary

    try:
        with lf.start_as_current_span(
            name="llm_judge_eval",
            metadata=run_metadata,
            input={"n_articles": len(rows)},
        ) as root:
            root.update_trace(
                name="LLM-as-judge summarization eval",
                session_id=session_id,
                tags=tags,
            )

            for row in rows:
                idx = row["article_index"]
                article = articles[idx]
                summary = summary_by_index.get(idx, "")

                with root.start_as_current_span(
                    name="evaluate_summary",
                    metadata={"article_index": idx},
                    input={
                        "title": _clip_text(article.title, 500),
                        "maintext": _clip_text(article.maintext, 8000),
                    },
                    output={
                        "summary": _clip_text(summary, 8000),
                        "overall_quality": row["overall_quality"],
                    },
                ) as art_span:
                    for metric in ("accuracy", "completeness", "conciseness", "clarity", "average_score"):
                        art_span.score(
                            name=metric,
                            value=float(row[metric]),
                            data_type="NUMERIC",
                        )

            root.update(output={"aggregates": aggregates})
            for metric_name in (
                "mean_accuracy",
                "mean_completeness",
                "mean_conciseness",
                "mean_clarity",
                "mean_average_score",
                "n_evaluated",
            ):
                root.score_trace(
                    name=metric_name,
                    value=float(aggregates[metric_name]),
                    data_type="NUMERIC",
                )
            trace_id_for_url = lf.get_current_trace_id()

        lf.flush()
        if trace_id_for_url:
            url = lf.get_trace_url(trace_id=trace_id_for_url)
            if url:
                logger.info("Langfuse LLM-judge trace: %s", url)
    except Exception:
        logger.exception("Failed to upload LLM-judge eval to Langfuse")


# ---------------------------------------------------------------------------
# Orchestrator helpers (called from run_workflow.py)
# ---------------------------------------------------------------------------


def run_entity_extraction_evals(
    results: list[Any],
    articles: list[Any],
    *,
    langfuse: bool = True,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all entity extraction evaluations and return results."""
    successful = sum(1 for r in results if r.error is None)
    console.print(
        f"\n[bold cyan]Running Entity Extraction Evaluations[/bold cyan] "
        f"({successful}/{len(results)} successful)"
    )
    return evaluate_entity_extraction(
        results, articles, langfuse=langfuse, run_context=run_context,
    )


async def run_summarization_evals(
    results: list[Any],
    articles: list[Any],
    *,
    langfuse: bool = True,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both similarity and LLM-as-judge evaluations on summaries."""
    successful = sum(1 for r in results if r.error is None)
    console.print(
        f"\n[bold cyan]Running Summarization Evaluations[/bold cyan] "
        f"({successful}/{len(results)} successful)"
    )

    similarity = evaluate_summarization_similarity(
        results, articles, langfuse=langfuse, run_context=run_context,
    )
    llm_judge = await evaluate_summarization_llm_judge(
        results, articles, langfuse=langfuse, run_context=run_context,
    )

    return {"similarity": similarity, "llm_judge": llm_judge}
