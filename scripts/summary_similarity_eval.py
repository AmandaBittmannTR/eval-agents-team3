"""Run the summarization agent on news articles and score outputs against ground truth.

Loads articles from ``data/transformed_data/*_data.csv`` (or a single such CSV), using columns
``title``, ``maintext``, and ``description``. Each row gets a stable id ``{csv_stem}:{row_index}``
(e.g. ``2020_data:0``) for joining agent outputs to reference text.

Calls :class:`~aieng.agent_evals.summarization.agent.SummarizationAgent` on each row
(``title`` + ``maintext``), then compares generated text to the dataset: cosine similarity
(TF--IDF) vs. ``description``, and BERTScore F1 with ``maintext`` as reference.

By default only **10** articles are processed (after sorting by id). Use ``-n`` / ``--limit`` for
another count, or ``--all`` to run the full dataset (slow).

API keys are read from the project ``.env`` at the repo root (not the process cwd), matching
other bootcamp scripts. Set ``GEMINI_API_KEY`` and/or ``OPENAI_API_KEY`` as in ``.env.example``;
if only ``OPENAI_API_KEY`` is set, it is mirrored to ``GEMINI_API_KEY`` for ADK.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


_REQUIRED_CSV_FIELDS = frozenset({"title", "maintext", "description"})


def discover_csv_files(data_path: Path) -> list[Path]:
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
    """Load all transformed CSV rows; each row has ``article_id``, ``title``, ``maintext``, ``description``."""
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
                        "description": (row.get("description") or "").strip(),
                    },
                )
    rows.sort(key=lambda r: r["article_id"])
    return rows


def ground_truth_from_article_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Build reference index from loaded CSV rows (same slice as summarization)."""
    return {
        r["article_id"]: {
            "description": r["description"],
            "maintext": r["maintext"],
            "title": r["title"],
        }
        for r in rows
    }


async def run_summarization_agent_batch(articles: list[dict[str, Any]]) -> dict[str, str]:
    """Run ``SummarizationAgent`` on articles; return map ``article_id`` → generated summary text."""
    from aieng.agent_evals.summarization.agent import SummarizationAgent

    if not articles:
        logger.warning("No articles to summarize.")
        return {}

    agent = SummarizationAgent()
    summaries: dict[str, str] = {}
    for i, row in enumerate(articles):
        article_id = row["article_id"]
        title = str(row.get("title") or "")
        body = str(row.get("maintext") or "")
        logger.info("Summarizing [%s/%s] %s", i + 1, len(articles), article_id)
        agent.reset()
        try:
            response = await agent.summarize_async(title=title, body=body)
            summary_text = response.text.strip()
        except Exception:
            logger.exception("Summarization failed for %s", article_id)
            summary_text = ""
        summaries[article_id] = summary_text
    return summaries


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two strings using TF--IDF bag-of-words vectors."""
    a, b = text_a.strip(), text_b.strip()
    if not a or not b:
        return 0.0
    vectorizer = TfidfVectorizer(strip_accents="unicode", min_df=1)
    matrix = vectorizer.fit_transform([a, b])
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
    return float(sim)


def iter_matched_pairs(
    ground: dict[str, dict[str, str]],
    agent: dict[str, str],
) -> Iterator[tuple[str, str, str, str]]:
    """Yield (article_id, agent_summary, reference_description, maintext)."""
    for article_id, summary in agent.items():
        if article_id not in ground:
            logger.warning("No ground-truth row for article_id (skipped): %s", article_id)
            continue
        g = ground[article_id]
        yield article_id, summary, g["description"], g["maintext"]


def run_eval(
    ground: dict[str, dict[str, str]],
    agent: dict[str, str],
    *,
    bert_model: str | None,
    device: str | None,
    batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute per-row metrics and aggregate means."""
    ids: list[str] = []
    summaries: list[str] = []
    descriptions: list[str] = []
    articles: list[str] = []

    for article_id, summary, desc, main in iter_matched_pairs(ground, agent):
        ids.append(article_id)
        summaries.append(summary)
        descriptions.append(desc)
        articles.append(main)

    if not ids:
        return [], {}

    cosine_scores = [tfidf_cosine_similarity(s, d) for s, d in zip(summaries, descriptions, strict=True)]

    kwargs: dict[str, Any] = {"lang": "en", "verbose": False}
    if bert_model:
        kwargs["model_type"] = bert_model
    if device:
        kwargs["device"] = device
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    # Candidate = generated summary, reference = source article (user-requested direction).
    _p, _r, f1 = bert_score(summaries, articles, **kwargs)
    bert_f1 = f1.tolist()

    rows: list[dict[str, Any]] = []
    for i, article_id in enumerate(ids):
        rows.append(
            {
                "article_id": article_id,
                "cosine_tfidf_vs_reference_summary": cosine_scores[i],
                "bertscore_f1_vs_article": bert_f1[i],
            },
        )

    aggregates = {
        "mean_cosine_tfidf_vs_reference_summary": sum(cosine_scores) / len(cosine_scores),
        "mean_bertscore_f1_vs_article": sum(bert_f1) / len(bert_f1),
        "n_matched": float(len(ids)),
    }
    return rows, aggregates


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
        help="Skip this many articles (after sorting by article_id) before summarizing (default: 0).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Summarize at most N articles after --offset (default: 10).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every article in the dataset (ignores --limit).",
    )
    parser.add_argument(
        "--bert-model",
        default=None,
        help="Optional BERTScore model name or path (default: English default in bert-score).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="BERTScore batch size (default: library default).",
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
    args = parser.parse_args(argv)
    article_limit: int | None = None if args.all else args.limit

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    # Configs() looks for ".env" relative to cwd; loading repo .env fixes runs from `scripts/`.
    load_dotenv(repo_root / ".env")
    # Mirror OpenAI key to GEMINI when only the OpenAI-compatible vars are set.
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    data_path = args.data if args.data.is_absolute() else repo_root / args.data

    csv_files = discover_csv_files(data_path)
    article_rows = load_csv_article_rows(csv_files)
    if args.offset:
        article_rows = article_rows[args.offset :]
    if article_limit is not None:
        article_rows = article_rows[:article_limit]

    agent_summaries = asyncio.run(run_summarization_agent_batch(article_rows))

    if not agent_summaries:
        logger.error("No summaries produced (empty dataset after offset/limit or all calls failed).")
        return 1

    ground = ground_truth_from_article_rows(article_rows)

    rows, aggregates = run_eval(
        ground,
        agent_summaries,
        bert_model=args.bert_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not rows:
        logger.error("No rows matched between produced summaries and ground-truth data.")
        return 1

    if not args.quiet:
        for row in rows:
            aid = row["article_id"]
            label = aid if len(aid) <= 70 else f"{aid[:67]}..."
            print(
                f"{label}\tcosine={row['cosine_tfidf_vs_reference_summary']:.4f}\t"
                f"bert_f1={row['bertscore_f1_vs_article']:.4f}",
            )

    print(json.dumps(aggregates, indent=2))

    if args.per_item_json:
        out_path = args.per_item_json if args.per_item_json.is_absolute() else repo_root / args.per_item_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
