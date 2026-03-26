"""Run the entity extraction agent on news articles and score outputs against ground truth.

Loads articles from ``data/transformed_data/*_data.csv`` (or a single such CSV), using columns
``title``, ``maintext``, ``mentioned_companies``, and ``named_entities``. Each row gets a stable
id ``{csv_stem}:{row_index}`` (e.g. ``2017_data:0``) for joining agent outputs to reference data.

Calls :func:`~aieng.agent_evals.entity_extraction.agent.run_entity_extraction` on each row
(``title`` + ``maintext``), then compares the extracted entities and ticker symbols against the
ground-truth columns using set-based precision, recall, and F1 metrics.

By default only **5** articles are processed (after sorting by id). Use ``-n`` / ``--limit`` for
another count, or ``--all`` to run the full dataset (slow).

API keys are read from the project ``.env`` at the repo root (not the process cwd), matching
other bootcamp scripts. Set ``GEMINI_API_KEY`` and/or ``OPENAI_API_KEY`` as in ``.env.example``;
if only ``OPENAI_API_KEY`` is set, it is mirrored to ``GEMINI_API_KEY`` for ADK.

Use ``--langfuse`` to send the same per-article and aggregate metrics to Langfuse as scores on a
single trace (requires ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and optional
``LANGFUSE_HOST`` in ``.env``).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_REQUIRED_CSV_FIELDS = frozenset({"title", "maintext", "mentioned_companies", "named_entities"})


# ---------------------------------------------------------------------------
# CSV discovery / loading
# ---------------------------------------------------------------------------


def discover_csv_files(data_path: Path) -> list[Path]:
    """Find one or more transformed-data CSVs at *data_path*."""
    if data_path.is_file():
        if data_path.suffix.lower() != ".csv":
            msg = f"Expected a .csv file: {data_path}"
            raise ValueError(msg)
        return [data_path]
    pattern = "*_data.csv"
    files = sorted(data_path.glob(pattern))
    if not files:
        msg = f"No {pattern} files under {data_path}"
        raise FileNotFoundError(msg)
    return files


def load_csv_article_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load all transformed CSV rows; each row has ``article_id`` plus the original columns."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        stem = path.stem
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            missing = _REQUIRED_CSV_FIELDS - fields
            if missing:
                msg = f"{path}: missing columns {sorted(missing)}"
                raise ValueError(msg)
            for i, row in enumerate(reader):
                main = (row.get("maintext") or "").strip()
                if not main:
                    continue
                article_id = f"{stem}:{i}"
                rows.append(
                    {
                        "article_id": article_id,
                        "title": (row.get("title") or "").strip(),
                        "maintext": main,
                        "mentioned_companies": row.get("mentioned_companies", ""),
                        "named_entities": row.get("named_entities", ""),
                    },
                )
    rows.sort(key=lambda r: r["article_id"])
    return rows


def ground_truth_from_article_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build reference index keyed by ``article_id``."""
    return {
        r["article_id"]: {
            "mentioned_companies": r["mentioned_companies"],
            "named_entities": r["named_entities"],
            "title": r["title"],
            "maintext": r["maintext"],
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Ground-truth parsing / normalization
# ---------------------------------------------------------------------------


def parse_mentioned_companies(raw: Any) -> set[str]:
    """Parse ``mentioned_companies`` from CSV string or list into uppercase ticker set."""
    if not raw:
        return set()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw in ("[]", "None"):
            return set()
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            logger.warning("Could not parse mentioned_companies: %s", raw[:200])
            return set()
    else:
        parsed = raw
    if isinstance(parsed, list):
        return {str(t).upper().strip() for t in parsed if t}
    return set()


def parse_named_entities(raw: Any) -> list[dict[str, Any]]:
    """Parse ``named_entities`` from CSV string, deduplicate by ``word`` (highest score wins)."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw in ("[]", "None"):
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            logger.warning("Could not parse named_entities: %s", raw[:200])
            return []
    else:
        parsed = raw
    if not isinstance(parsed, list):
        return []

    best_by_word: dict[str, dict[str, Any]] = {}
    for ent in parsed:
        if not isinstance(ent, dict):
            continue
        word = ent.get("word", "")
        if not word:
            continue
        score = ent.get("score", 0.0) or 0.0
        existing = best_by_word.get(word)
        if existing is None or score > existing.get("_score", 0.0):
            best_by_word[word] = {
                "entity_group": ent.get("entity_group", "MISC"),
                "word": word,
                "normalized": ent.get("normalized"),
                "_score": score,
            }
    return [
        {"entity_group": e["entity_group"], "word": e["word"], "normalized": e["normalized"]}
        for e in best_by_word.values()
    ]


_NORM_PUNCT_RE = re.compile(r"[.\,;:!?'\"()\[\]{}/\\]+")
_NORM_SPACE_RE = re.compile(r"\s+")


def _norm_word(raw: str) -> str:
    """Normalize an entity word for comparison.

    Handles BERT ``##`` subword artifacts, stray punctuation between tokens
    (e.g. ``Alphabet. Inc`` → ``alphabet inc``), and collapses whitespace.
    """
    w = raw.replace("##", "")
    w = _NORM_PUNCT_RE.sub(" ", w)
    w = _NORM_SPACE_RE.sub(" ", w).strip().lower()
    return w


def normalize_entity_set(entities: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Set of ``(entity_group, normalized_word)`` tuples for strict comparison."""
    result: set[tuple[str, str]] = set()
    for e in entities:
        w = _norm_word(str(e.get("word", "")))
        if w:
            result.add((str(e.get("entity_group", "MISC")).upper(), w))
    return result


def normalize_entity_set_relaxed(entities: list[dict[str, Any]]) -> set[str]:
    """Set of normalized words only (group-agnostic) for relaxed entity comparison."""
    return {_norm_word(str(e.get("word", ""))) for e in entities if e.get("word")}


def normalize_word_set(entities: list[dict[str, Any]]) -> set[str]:
    """Set of normalized entity words, ignoring entity_group."""
    return {_norm_word(str(e.get("word", ""))) for e in entities if e.get("word")}


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; returns 0.0 when both are zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


async def run_entity_extraction_batch(
    articles: list[dict[str, Any]],
    concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    """Run ``run_entity_extraction`` on each article; return ``article_id -> output dict``.

    Parameters
    ----------
    concurrency : int
        Max number of articles to process in parallel.  ``1`` (default) means
        sequential execution.
    """
    from aieng.agent_evals.entity_extraction.agent import run_entity_extraction

    if not articles:
        logger.warning("No articles to process.")
        return {}

    results: dict[str, dict[str, Any]] = {}
    sem = asyncio.Semaphore(concurrency)
    total = len(articles)
    counter = {"done": 0}

    async def _process(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        article_id = row["article_id"]
        title = str(row.get("title") or "")
        maintext = str(row.get("maintext") or "")
        async with sem:
            try:
                result = await run_entity_extraction(title=title, maintext=maintext)
                out = result.model_dump(mode="json")
            except Exception:
                logger.exception("Entity extraction failed for %s", article_id)
                out = {"mentioned_companies": [], "named_entities": []}
            counter["done"] += 1
            logger.info("Extracted [%s/%s] %s", counter["done"], total, article_id)
            results[article_id] = out
            return article_id, out

    tasks = [_process(row) for row in articles]
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning(
            "Batch interrupted after %d/%d articles. Returning partial results.",
            len(results),
            total,
        )

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _score_item(
    output: dict[str, Any],
    ground: dict[str, str],
) -> dict[str, Any]:
    """Compute per-item metrics comparing agent output to ground truth."""
    predicted_companies = {
        str(t).upper().strip()
        for t in (output.get("mentioned_companies") or [])
        if t
    }
    expected_companies = parse_mentioned_companies(ground.get("mentioned_companies"))

    co_tp = predicted_companies & expected_companies
    co_fp = predicted_companies - expected_companies
    co_fn = expected_companies - predicted_companies
    co_precision = len(co_tp) / len(predicted_companies) if predicted_companies else 0.0
    co_recall = len(co_tp) / len(expected_companies) if expected_companies else 0.0
    co_f1 = _f1(co_precision, co_recall)

    logger.debug(
        "Companies | predicted=%s | expected=%s | tp=%s | fp=%s | fn=%s | P=%.3f R=%.3f F1=%.3f",
        sorted(predicted_companies), sorted(expected_companies),
        sorted(co_tp), sorted(co_fp), sorted(co_fn),
        co_precision, co_recall, co_f1,
    )

    predicted_entities = output.get("named_entities") or []
    expected_entities = parse_named_entities(ground.get("named_entities"))

    # Strict entity match: (entity_group, word) tuples
    predicted_entity_set = normalize_entity_set(predicted_entities)
    expected_entity_set = normalize_entity_set(expected_entities)

    ent_tp = predicted_entity_set & expected_entity_set
    ent_fp = predicted_entity_set - expected_entity_set
    ent_fn = expected_entity_set - predicted_entity_set
    ent_precision = len(ent_tp) / len(predicted_entity_set) if predicted_entity_set else 0.0
    ent_recall = len(ent_tp) / len(expected_entity_set) if expected_entity_set else 0.0
    ent_f1 = _f1(ent_precision, ent_recall)

    logger.debug(
        "Strict entities | predicted=%s | expected=%s | tp=%s | fp=%s | fn=%s | P=%.3f R=%.3f F1=%.3f",
        sorted(predicted_entity_set), sorted(expected_entity_set),
        sorted(ent_tp), sorted(ent_fp), sorted(ent_fn),
        ent_precision, ent_recall, ent_f1,
    )

    # Relaxed entity match: words only (group-agnostic)
    pred_relaxed = normalize_entity_set_relaxed(predicted_entities)
    exp_relaxed = normalize_entity_set_relaxed(expected_entities)
    rel_tp = pred_relaxed & exp_relaxed
    rel_fp = pred_relaxed - exp_relaxed
    rel_fn = exp_relaxed - pred_relaxed
    rel_precision = len(rel_tp) / len(pred_relaxed) if pred_relaxed else 0.0
    rel_recall = len(rel_tp) / len(exp_relaxed) if exp_relaxed else 0.0
    rel_f1 = _f1(rel_precision, rel_recall)

    logger.debug(
        "Relaxed entities | predicted=%s | expected=%s | tp=%s | fp=%s | fn=%s | P=%.3f R=%.3f F1=%.3f",
        sorted(pred_relaxed), sorted(exp_relaxed),
        sorted(rel_tp), sorted(rel_fp), sorted(rel_fn),
        rel_precision, rel_recall, rel_f1,
    )

    # Word-level overlap (individual words from multi-word entities)
    predicted_words = normalize_word_set(predicted_entities)
    expected_words = normalize_word_set(expected_entities)
    word_tp = predicted_words & expected_words
    word_precision = len(word_tp) / len(predicted_words) if predicted_words else 0.0
    word_recall = len(word_tp) / len(expected_words) if expected_words else 0.0
    word_f1 = _f1(word_precision, word_recall)

    logger.debug(
        "Word overlap | predicted=%s | expected=%s | matched=%s | P=%.3f R=%.3f F1=%.3f",
        sorted(predicted_words), sorted(expected_words),
        sorted(word_tp),
        word_precision, word_recall, word_f1,
    )

    return {
        "companies_precision": co_precision,
        "companies_recall": co_recall,
        "companies_f1": co_f1,
        "companies_tp": len(co_tp),
        "companies_fp": len(co_fp),
        "companies_fn": len(co_fn),
        "entities_precision": ent_precision,
        "entities_recall": ent_recall,
        "entities_f1": ent_f1,
        "entities_tp": len(ent_tp),
        "entities_fp": len(ent_fp),
        "entities_fn": len(ent_fn),
        "entities_relaxed_precision": rel_precision,
        "entities_relaxed_recall": rel_recall,
        "entities_relaxed_f1": rel_f1,
        "entities_relaxed_tp": len(rel_tp),
        "entities_relaxed_fp": len(rel_fp),
        "entities_relaxed_fn": len(rel_fn),
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
        "word_matched": len(word_tp),
        "word_expected": len(expected_words),
        "word_predicted": len(predicted_words),
    }


def iter_matched_pairs(
    ground: dict[str, dict[str, str]],
    agent: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, dict[str, Any], dict[str, str]]]:
    """Yield ``(article_id, agent_output, ground_truth_row)``."""
    for article_id, output in agent.items():
        if article_id not in ground:
            logger.warning("No ground-truth row for article_id (skipped): %s", article_id)
            continue
        yield article_id, output, ground[article_id]


def run_eval(
    ground: dict[str, dict[str, str]],
    agent: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute per-row metrics and aggregate means."""
    per_item: list[dict[str, Any]] = []

    co_tp_total = co_fp_total = co_fn_total = 0
    ent_tp_total = ent_fp_total = ent_fn_total = 0
    rel_tp_total = rel_fp_total = rel_fn_total = 0
    word_matched_total = word_expected_total = word_predicted_total = 0

    for article_id, output, gt in iter_matched_pairs(ground, agent):
        scores = _score_item(output, gt)
        scores["article_id"] = article_id
        per_item.append(scores)

        co_tp_total += scores["companies_tp"]
        co_fp_total += scores["companies_fp"]
        co_fn_total += scores["companies_fn"]
        ent_tp_total += scores["entities_tp"]
        ent_fp_total += scores["entities_fp"]
        ent_fn_total += scores["entities_fn"]
        rel_tp_total += scores["entities_relaxed_tp"]
        rel_fp_total += scores["entities_relaxed_fp"]
        rel_fn_total += scores["entities_relaxed_fn"]
        word_matched_total += scores["word_matched"]
        word_expected_total += scores["word_expected"]
        word_predicted_total += scores["word_predicted"]

    if not per_item:
        return [], {}

    n = len(per_item)
    macro_co_prec = co_tp_total / (co_tp_total + co_fp_total) if (co_tp_total + co_fp_total) else 0.0
    macro_co_recall = co_tp_total / (co_tp_total + co_fn_total) if (co_tp_total + co_fn_total) else 0.0
    macro_ent_prec = ent_tp_total / (ent_tp_total + ent_fp_total) if (ent_tp_total + ent_fp_total) else 0.0
    macro_ent_recall = ent_tp_total / (ent_tp_total + ent_fn_total) if (ent_tp_total + ent_fn_total) else 0.0
    macro_rel_prec = rel_tp_total / (rel_tp_total + rel_fp_total) if (rel_tp_total + rel_fp_total) else 0.0
    macro_rel_recall = rel_tp_total / (rel_tp_total + rel_fn_total) if (rel_tp_total + rel_fn_total) else 0.0
    macro_word_prec = word_matched_total / word_predicted_total if word_predicted_total else 0.0
    macro_word_recall = word_matched_total / word_expected_total if word_expected_total else 0.0

    aggregates: dict[str, float] = {
        "n_matched": float(n),
        "avg_companies_f1": sum(r["companies_f1"] for r in per_item) / n,
        "avg_entities_f1": sum(r["entities_f1"] for r in per_item) / n,
        "avg_entities_relaxed_f1": sum(r["entities_relaxed_f1"] for r in per_item) / n,
        "avg_word_f1": sum(r["word_f1"] for r in per_item) / n,
        "macro_companies_precision": macro_co_prec,
        "macro_companies_recall": macro_co_recall,
        "macro_companies_f1": _f1(macro_co_prec, macro_co_recall),
        "macro_entities_precision": macro_ent_prec,
        "macro_entities_recall": macro_ent_recall,
        "macro_entities_f1": _f1(macro_ent_prec, macro_ent_recall),
        "macro_entities_relaxed_precision": macro_rel_prec,
        "macro_entities_relaxed_recall": macro_rel_recall,
        "macro_entities_relaxed_f1": _f1(macro_rel_prec, macro_rel_recall),
        "macro_word_precision": macro_word_prec,
        "macro_word_recall": macro_word_recall,
        "macro_word_f1": _f1(macro_word_prec, macro_word_recall),
    }
    return per_item, aggregates


# ---------------------------------------------------------------------------
# Langfuse upload
# ---------------------------------------------------------------------------


def _clip_text(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return f"{t[: max_chars - 3]}..."


def push_entity_extraction_eval_to_langfuse(
    *,
    rows: list[dict[str, Any]],
    aggregates: dict[str, float],
    ground: dict[str, dict[str, str]],
    agent_outputs: dict[str, dict[str, Any]],
    run_metadata: dict[str, Any],
) -> None:
    """Send one Langfuse trace with per-article spans, scores, and aggregate trace scores."""
    from aieng.agent_evals.async_client_manager import AsyncClientManager

    manager = AsyncClientManager.get_instance()
    cfg = manager.configs
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must both be set; skipping Langfuse upload.",
        )
        return

    lf = manager.langfuse_client
    try:
        if not lf.auth_check():
            logger.warning("Langfuse authentication failed; skipping upload.")
            return
    except Exception:
        logger.exception("Langfuse auth_check failed; skipping upload.")
        return

    session_id = run_metadata.get(
        "run_id", f"entity_extraction_eval-{uuid.uuid4().hex[:12]}"
    )
    tags = ["entity_extraction_eval", "bootcamp"]
    if run_metadata.get("source") == "workflow":
        tags.append("Full Workflow Pipeline")
    trace_id_for_url: str | None = None

    try:
        with lf.start_as_current_span(
            name="entity_extraction_eval",
            metadata=run_metadata,
            input={
                "data_path": run_metadata.get("data_path"),
                "offset": run_metadata.get("offset"),
                "limit": run_metadata.get("limit"),
                "all": run_metadata.get("all"),
                "n_articles": len(rows),
            },
        ) as root:
            root.update_trace(
                name="Entity extraction eval",
                session_id=session_id,
                tags=tags,
            )
            for row in rows:
                aid = row["article_id"]
                ref = ground[aid]
                output = agent_outputs.get(aid, {})
                with root.start_as_current_span(
                    name="extract_entities",
                    metadata={"article_id": aid},
                    input={
                        "article_id": aid,
                        "title": _clip_text(ref["title"], 500),
                        "maintext": _clip_text(ref["maintext"], 8000),
                    },
                    output={
                        "mentioned_companies": output.get("mentioned_companies", []),
                        "named_entities": output.get("named_entities", []),
                    },
                ) as art_span:
                    art_span.update(
                        metadata={
                            "expected_companies": _clip_text(ref["mentioned_companies"], 2000),
                            "expected_entities": _clip_text(ref["named_entities"], 4000),
                        },
                    )
                    art_span.score(
                        name="companies_f1",
                        value=float(row["companies_f1"]),
                        data_type="NUMERIC",
                    )
                    art_span.score(
                        name="entities_f1",
                        value=float(row["entities_f1"]),
                        data_type="NUMERIC",
                    )
                    art_span.score(
                        name="word_f1",
                        value=float(row["word_f1"]),
                        data_type="NUMERIC",
                    )

            root.update(output={"aggregates": aggregates})
            for metric_name in (
                "avg_companies_f1",
                "avg_entities_f1",
                "avg_word_f1",
                "macro_companies_f1",
                "macro_entities_f1",
                "macro_word_f1",
                "n_matched",
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
                logger.info("Langfuse trace: %s", url)
    except Exception:
        logger.exception("Failed to upload entity extraction eval to Langfuse")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/transformed_data"),
        help="Directory containing *_data.csv or a single such CSV file.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many articles (after sorting by article_id) before processing (default: 0).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Process at most N articles after --offset (default: 5).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every article in the dataset (ignores --limit).",
    )
    parser.add_argument(
        "--per-item-json",
        type=Path,
        default=None,
        help="If set, write one JSON object per line with per-article scores.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print aggregate JSON, not per-row table.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max articles to process in parallel (default: 1 = sequential).",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help=(
            "Upload this run to Langfuse: one trace with per-article spans/scores and aggregate "
            "trace scores (requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging to show per-article scoring details (predicted vs expected).",
    )
    args = parser.parse_args(argv)
    article_limit: int | None = None if args.all else args.limit

    if args.quiet:
        log_level = logging.WARNING
    elif args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    data_path = args.data if args.data.is_absolute() else repo_root / args.data

    csv_files = discover_csv_files(data_path)
    article_rows = load_csv_article_rows(csv_files)
    if args.offset:
        article_rows = article_rows[args.offset:]
    if article_limit is not None:
        article_rows = article_rows[:article_limit]

    logger.info(
        "Running entity extraction on %d article(s) (concurrency=%d)...",
        len(article_rows),
        args.concurrency,
    )

    agent_outputs: dict[str, dict[str, Any]] = {}
    interrupted = False
    try:
        agent_outputs = asyncio.run(
            run_entity_extraction_batch(article_rows, concurrency=args.concurrency),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted! Writing partial results...")
        interrupted = True
    except Exception:
        logger.exception("Batch extraction failed. Writing partial results...")
        interrupted = True

    def _report_results() -> int:
        """Evaluate whatever outputs we have and write results to console/Langfuse."""
        if not agent_outputs:
            logger.error(
                "No outputs produced (empty dataset after offset/limit or all calls failed).",
            )
            return 1

        ground = ground_truth_from_article_rows(article_rows)
        per_item, aggregates = run_eval(ground, agent_outputs)

        if not per_item:
            logger.error("No rows matched between agent outputs and ground-truth data.")
            return 1

        if interrupted:
            aggregates["partial_run"] = True
            aggregates["articles_completed"] = float(len(agent_outputs))
            aggregates["articles_requested"] = float(len(article_rows))

        if not args.quiet:
            for row in per_item:
                aid = row["article_id"]
                label = aid if len(aid) <= 50 else f"{aid[:47]}..."
                print(
                    f"{label}\t"
                    f"co_f1={row['companies_f1']:.4f}\t"
                    f"ent_f1={row['entities_f1']:.4f}\t"
                    f"ent_rel_f1={row['entities_relaxed_f1']:.4f}\t"
                    f"word_f1={row['word_f1']:.4f}",
                )

        print(json.dumps(aggregates, indent=2))

        if args.per_item_json:
            out_path = (
                args.per_item_json
                if args.per_item_json.is_absolute()
                else repo_root / args.per_item_json
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for row in per_item:
                    f.write(json.dumps(row) + "\n")
            logger.info("Per-item scores written to %s", out_path)

        if args.langfuse:
            push_entity_extraction_eval_to_langfuse(
                rows=per_item,
                aggregates=aggregates,
                ground=ground,
                agent_outputs=agent_outputs,
                run_metadata={
                    "data_path": str(data_path),
                    "offset": args.offset,
                    "limit": None if args.all else args.limit,
                    "all": args.all,
                    "interrupted": interrupted,
                },
            )

        return 1 if interrupted else 0

    return _report_results()


if __name__ == "__main__":
    sys.exit(main())
