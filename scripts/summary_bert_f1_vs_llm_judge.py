"""Run summarization + BERTScore F1 (same as ``summary_similarity_eval``) and LLM-as-judge on the same rows.

Loads articles with the same rules as ``scripts/summary_similarity_eval.py`` (sorted ``article_id``,
``--offset`` / ``--limit``). For each row:

- Generates a summary with ``SummarizationAgent``.
- Computes ``bertscore_f1_vs_article`` (candidate = summary, reference = full article).
- Calls ``create_llm_as_judge_evaluator`` with the default rubric (correctness, completeness,
  constraint_adherence). **Overall judge score** = mean of those three metrics (0/1 coerced to float).

Writes an interactive Plotly scatter: x = BERTScore F1, y = overall LLM judge score.

Use ``--langfuse`` to log a trace with per-article scores and attach the scatter plot as **PNG media**
on the root span output (requires ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and optional
``LANGFUSE_HOST``). PNG export uses Plotly + **kaleido** (declared in project dependencies).

Environment: same as the bootcamp ``.env`` at repo root (``GEMINI_API_KEY`` / ``OPENAI_API_KEY``).

Example::

    uv run python scripts/summary_bert_f1_vs_llm_judge.py -n 50 --plot outputs/bert_vs_judge.html --langfuse
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dotenv import load_dotenv
from langfuse.experiment import Evaluation
from langfuse.media import LangfuseMedia

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

RUBRIC_METRICS = ("correctness", "completeness", "constraint_adherence")


def _load_summary_similarity_eval_module() -> Any:
    path = Path(__file__).resolve().parent / "summary_similarity_eval.py"
    spec = importlib.util.spec_from_file_location("summary_similarity_eval", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load module from {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip_text(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return f"{t[: max_chars - 3]}..."


def _metric_to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes"):
            return 1.0
        if v in ("0", "false", "no"):
            return 0.0
    return None


def overall_llm_judge_score(evals: list[Evaluation]) -> tuple[float | None, dict[str, Any]]:
    """Return mean of default rubric metrics, or None if judge failed."""
    by_name: dict[str, Any] = {e.name: e.value for e in evals}
    if "llm_judge_error" in by_name:
        return None, by_name
    vals: list[float] = []
    for name in RUBRIC_METRICS:
        if name not in by_name:
            continue
        f = _metric_to_float(by_name[name])
        if f is not None:
            vals.append(f)
    if not vals:
        return None, by_name
    return sum(vals) / len(vals), by_name


async def run_llm_judge_batch(
    *,
    article_rows: list[dict[str, Any]],
    summaries: dict[str, str],
    max_body_chars: int,
    concurrency: int,
    judge_config: LLMRequestConfig | None,
) -> dict[str, tuple[float | None, dict[str, Any], list[Evaluation]]]:
    evaluator = create_llm_as_judge_evaluator(
        name="llm_judge",
        model_config=judge_config,
    )
    sem = asyncio.Semaphore(max(1, concurrency))
    results: dict[str, tuple[float | None, dict[str, Any], list[Evaluation]]] = {}

    async def one(row: dict[str, Any]) -> None:
        aid = row["article_id"]
        summary = summaries.get(aid, "")
        async with sem:
            inp = {
                "instruction": (
                    "Summarize the news article below. The summary should capture the main points "
                    "in concise language."
                ),
                "title": row.get("title") or "",
                "article_body": _clip_text(str(row.get("maintext") or ""), max_body_chars),
            }
            expected = str(row.get("description") or "")
            try:
                evs = await evaluator(
                    input=inp,
                    output=summary,
                    expected_output=expected,
                    metadata={"article_id": aid},
                )
            except Exception:
                logger.exception("LLM judge failed for %s", aid)
                results[aid] = (None, {"error": "exception"}, [])
                return
            overall, detail = overall_llm_judge_score(evs)
            results[aid] = (overall, detail, evs)

    await asyncio.gather(*[one(r) for r in article_rows])
    return results


def make_figure(points: list[dict[str, Any]]) -> go.Figure:
    ok = [p for p in points if p.get("llm_judge_overall") is not None]

    fig = go.Figure()
    if ok:
        fig.add_trace(
            go.Scatter(
                x=[p["bertscore_f1_vs_article"] for p in ok],
                y=[p["llm_judge_overall"] for p in ok],
                mode="markers",
                name="items",
                text=[p["article_id"] for p in ok],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "BERTScore F1: %{x:.4f}<br>"
                    "LLM judge (mean): %{y:.4f}<extra></extra>"
                ),
            ),
        )
    else:
        fig.add_annotation(
            text="No successful LLM judge scores — check logs and --merged-jsonl",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig.update_layout(
        title="BERTScore F1 (summary vs article) vs mean LLM-judge rubric score",
        xaxis_title="BERTScore F1 (vs article)",
        yaxis_title="Mean(correctness, completeness, constraint_adherence)",
        yaxis=dict(range=[-0.05, 1.05]),
        legend_title="",
        template="plotly_white",
    )
    return fig


def write_plot_html(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info("Wrote plot to %s", out_path)


def figure_to_png_bytes(fig: go.Figure, *, width: int = 960, height: int = 640) -> bytes | None:
    """Render Plotly figure to PNG for Langfuse media upload (requires kaleido)."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1)
    except Exception as exc:
        logger.warning(
            "Could not export plot to PNG (install/sync kaleido): %s",
            exc,
        )
        return None


def _span_score_evaluation(span: Any, ev: Evaluation) -> None:
    """Attach one Langfuse span score from an LLM-judge ``Evaluation``."""
    if ev.value is None:
        return
    v = ev.value
    comment = ev.comment
    if isinstance(v, bool):
        span.score(name=ev.name, value=1.0 if v else 0.0, data_type="NUMERIC", comment=comment)
        return
    if isinstance(v, (int, float)):
        span.score(name=ev.name, value=float(v), data_type="NUMERIC", comment=comment)
        return
    if isinstance(v, str):
        span.score(name=ev.name, value=v, data_type="CATEGORICAL", comment=comment)


def push_bert_vs_judge_to_langfuse(
    *,
    points: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    judge_by_id: dict[str, tuple[float | None, dict[str, Any], list[Evaluation]]],
    aggregates: dict[str, float],
    fig: go.Figure,
    pearson_r: float | None,
    run_metadata: dict[str, Any],
    summary_for_output: dict[str, Any],
) -> None:
    """One Langfuse trace: per-article spans, aggregate scores, scatter plot as PNG on root output."""
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

    png = figure_to_png_bytes(fig)
    session_id = f"bert_f1_vs_llm_judge-{uuid.uuid4().hex[:12]}"
    trace_id_for_url: str | None = None
    metrics_by_aid = {r["article_id"]: r for r in metric_rows}

    try:
        with lf.start_as_current_span(
            name="bert_f1_vs_llm_judge_eval",
            metadata=run_metadata,
            input={
                "data_path": run_metadata.get("data_path"),
                "offset": run_metadata.get("offset"),
                "limit": run_metadata.get("limit"),
                "all": run_metadata.get("all"),
                "n_articles": len(points),
            },
        ) as root:
            root.update_trace(
                name="BERT F1 vs LLM judge",
                session_id=session_id,
                tags=["bert_f1_vs_llm_judge", "bootcamp", "summarization"],
            )
            for p in points:
                aid = p["article_id"]
                mrow = metrics_by_aid.get(aid, {})
                overall, _detail, evs = judge_by_id.get(aid, (None, {}, []))
                with root.start_as_current_span(
                    name="summarize_and_score_article",
                    metadata={"article_id": aid},
                    input={"article_id": aid},
                    output={"summary_metrics": {"llm_judge_overall": overall}},
                ) as art_span:
                    art_span.score(
                        name="bertscore_f1_vs_article",
                        value=float(p["bertscore_f1_vs_article"]),
                        data_type="NUMERIC",
                    )
                    art_span.score(
                        name="cosine_tfidf_vs_reference_summary",
                        value=float(mrow.get("cosine_tfidf_vs_reference_summary", 0.0)),
                        data_type="NUMERIC",
                    )
                    if overall is not None:
                        art_span.score(
                            name="llm_judge_overall",
                            value=float(overall),
                            data_type="NUMERIC",
                        )
                    for ev in evs:
                        _span_score_evaluation(art_span, ev)

            out_payload: dict[str, Any] = {"run_summary": summary_for_output}
            if png is not None:
                out_payload["scatter_plot_bert_f1_vs_llm_judge"] = LangfuseMedia(
                    content_bytes=png,
                    content_type="image/png",
                )
            root.update(output=out_payload)

            root.score_trace(
                name="mean_bertscore_f1_vs_article",
                value=float(aggregates["mean_bertscore_f1_vs_article"]),
                data_type="NUMERIC",
            )
            root.score_trace(
                name="mean_cosine_tfidf_vs_reference_summary",
                value=float(aggregates["mean_cosine_tfidf_vs_reference_summary"]),
                data_type="NUMERIC",
            )
            root.score_trace(
                name="n_matched",
                value=float(aggregates["n_matched"]),
                data_type="NUMERIC",
            )
            if summary_for_output.get("mean_llm_judge_overall") is not None:
                root.score_trace(
                    name="mean_llm_judge_overall",
                    value=float(summary_for_output["mean_llm_judge_overall"]),
                    data_type="NUMERIC",
                )
            root.score_trace(
                name="n_llm_judge_ok",
                value=float(summary_for_output.get("n_judged_ok", 0)),
                data_type="NUMERIC",
            )
            if pearson_r is not None and pearson_r == pearson_r:  # not NaN
                root.score_trace(
                    name="pearson_r_bert_f1_vs_llm_judge",
                    value=float(pearson_r),
                    data_type="NUMERIC",
                )
            trace_id_for_url = lf.get_current_trace_id()

        lf.flush()
        if trace_id_for_url:
            url = lf.get_trace_url(trace_id=trace_id_for_url)
            if url:
                logger.info("Langfuse trace: %s", url)
    except Exception:
        logger.exception("Failed to upload bert_f1_vs_llm_judge eval to Langfuse")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/transformed_data"),
        help="Directory containing *_data.csv or a single CSV (same as summary_similarity_eval).",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip N articles after sorting by id.")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Process N articles (default: 50).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process entire dataset (ignores --limit).",
    )
    parser.add_argument("--bert-model", default=None, help="Optional BERTScore model (bert-score).")
    parser.add_argument("--device", default=None, help="Torch device for BERTScore.")
    parser.add_argument("--batch-size", type=int, default=None, help="BERTScore batch size.")
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=12_000,
        help="Max chars of article body passed to the LLM judge input (default: 12000).",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=3,
        help="Concurrent LLM judge API calls (default: 3).",
    )
    parser.add_argument(
        "--judge-timeout-sec",
        type=float,
        default=120.0,
        help="Timeout per judge request in seconds (default: 120).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("outputs/bert_f1_vs_llm_judge.html"),
        help="Output HTML path for the Plotly figure.",
    )
    parser.add_argument(
        "--merged-jsonl",
        type=Path,
        default=None,
        help="If set, write one JSON object per line (scores + judge breakdown).",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help=(
            "Upload one trace to Langfuse: per-article scores, trace-level aggregates, "
            "and the scatter plot as PNG on the root span output (needs Langfuse keys in .env)."
        ),
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Less logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    load_dotenv(_REPO_ROOT / ".env")
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    sse = _load_summary_similarity_eval_module()
    data_path = args.data if args.data.is_absolute() else _REPO_ROOT / args.data
    article_limit: int | None = None if args.all else args.limit

    csv_files = sse.discover_csv_files(data_path)
    article_rows = sse.load_csv_article_rows(csv_files)
    if args.offset:
        article_rows = article_rows[args.offset :]
    if article_limit is not None:
        article_rows = article_rows[:article_limit]

    if not article_rows:
        logger.error("No articles after offset/limit.")
        return 1

    async def _run_pipeline() -> tuple[dict[str, str], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
        summaries_local = await sse.run_summarization_agent_batch(article_rows)
        ground_local = sse.ground_truth_from_article_rows(article_rows)
        rows_local, ag_local = sse.run_eval(
            ground_local,
            summaries_local,
            bert_model=args.bert_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        judge_cfg = LLMRequestConfig(timeout_sec=args.judge_timeout_sec)
        try:
            judge_local = await run_llm_judge_batch(
                article_rows=article_rows,
                summaries=summaries_local,
                max_body_chars=args.max_body_chars,
                concurrency=args.judge_concurrency,
                judge_config=judge_cfg,
            )
        finally:
            await AsyncClientManager.get_instance().close()
        return summaries_local, rows_local, ag_local, judge_local

    summaries, metric_rows, aggregates, judge_by_id = asyncio.run(_run_pipeline())
    if not summaries:
        logger.error("No summaries produced.")
        return 1
    if not metric_rows:
        logger.error("No scored rows from run_eval.")
        return 1

    points: list[dict[str, Any]] = []
    for row in metric_rows:
        aid = row["article_id"]
        overall, detail, _evs = judge_by_id.get(aid, (None, {}, []))
        rec = {
            "article_id": aid,
            "bertscore_f1_vs_article": float(row["bertscore_f1_vs_article"]),
            "cosine_tfidf_vs_reference_summary": float(row["cosine_tfidf_vs_reference_summary"]),
            "llm_judge_overall": overall,
            "llm_judge_metrics": detail,
        }
        points.append(rec)

    pearson_r: float | None = None
    xs = [p["bertscore_f1_vs_article"] for p in points if p["llm_judge_overall"] is not None]
    ys = [p["llm_judge_overall"] for p in points if p["llm_judge_overall"] is not None]
    if len(xs) >= 2:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
        den_y = sum((y - my) ** 2 for y in ys) ** 0.5
        r = num / (den_x * den_y) if den_x > 0 and den_y > 0 else float("nan")
        pearson_r = r if r == r else None  # exclude NaN
        logger.info("Pearson r (BERT F1 vs mean judge): %.4f (n=%s)", r, len(xs))

    out_plot = args.plot if args.plot.is_absolute() else _REPO_ROOT / args.plot
    fig = make_figure(points)
    write_plot_html(fig, out_plot)

    n_ok = sum(1 for p in points if p["llm_judge_overall"] is not None)
    summary_out = {
        "mean_bertscore_f1_vs_article": aggregates.get("mean_bertscore_f1_vs_article"),
        "mean_llm_judge_overall": (
            sum(p["llm_judge_overall"] for p in points if p["llm_judge_overall"] is not None) / n_ok
            if n_ok
            else None
        ),
        "n_judged_ok": n_ok,
        "n_judge_failed": len(points) - n_ok,
        "n_total": len(points),
        "pearson_r_bert_f1_vs_llm_judge": pearson_r,
        "plot": str(out_plot),
    }
    print(json.dumps(summary_out, indent=2))

    if args.langfuse:
        push_bert_vs_judge_to_langfuse(
            points=points,
            metric_rows=metric_rows,
            judge_by_id=judge_by_id,
            aggregates=aggregates,
            fig=fig,
            pearson_r=pearson_r,
            run_metadata={
                "data_path": str(data_path),
                "offset": args.offset,
                "limit": None if args.all else args.limit,
                "all": args.all,
                "bert_model": args.bert_model,
                "judge_concurrency": args.judge_concurrency,
            },
            summary_for_output=summary_out,
        )

    if args.merged_jsonl:
        mj = args.merged_jsonl if args.merged_jsonl.is_absolute() else _REPO_ROOT / args.merged_jsonl
        mj.parent.mkdir(parents=True, exist_ok=True)
        with mj.open("w", encoding="utf-8") as f:
            for rec in points:
                f.write(json.dumps(rec, default=str) + "\n")
        logger.info("Wrote %s", mj)

    return 0


if __name__ == "__main__":
    sys.exit(main())
