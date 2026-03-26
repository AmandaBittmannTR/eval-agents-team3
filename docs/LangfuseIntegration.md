# Langfuse integration in this project

This document summarizes **how data leaves the codebase and reaches Langfuse**: configuration, transports, and the main entry points.

## Configuration

Settings live in `Configs` (`aieng-eval-agents/aieng/agent_evals/configs.py`) and load from environment variables and `.env` (via Pydantic Settings).

| Setting | Environment variable | Notes |
|--------|----------------------|--------|
| `langfuse_public_key` | `LANGFUSE_PUBLIC_KEY` | Must match pattern `pk-lf-...` |
| `langfuse_secret_key` | `LANGFUSE_SECRET_KEY` | Validated to start with `sk-lf-` |
| `langfuse_host` | `LANGFUSE_HOST` | Default `https://us.cloud.langfuse.com` |

The shared **Langfuse SDK client** is created lazily by `AsyncClientManager.langfuse_client` (`aieng-eval-agents/aieng/agent_evals/async_client_manager.py`): it instantiates `Langfuse(public_key=..., secret_key=..., host=...)`. That client is what most programmatic calls (datasets, scores, trace reads, flush) use.

## How data is sent (two main transports)

### 1. Traces via OpenTelemetry (OTLP HTTP)

**Purpose:** Ship **spans/traces** from running agents (especially Google ADK) to Langfuse’s OTLP endpoint.

**Implementation:** `init_tracing()` in `aieng-eval-agents/aieng/agent_evals/langfuse.py`.

Flow in short:

1. Obtains the singleton `AsyncClientManager`, builds the Langfuse client, and runs `auth_check()`.
2. Sets `OTEL_EXPORTER_OTLP_ENDPOINT` to `{langfuse_host}/api/public/otel` and `OTEL_EXPORTER_OTLP_HEADERS` to `Authorization=Basic <base64(public_key:secret_key)>`.
3. Registers a global `TracerProvider` with `BatchSpanProcessor` + `OTLPSpanExporter` pointing at `{otel_endpoint}/v1/traces` with the same Basic auth header.
4. Calls `GoogleADKInstrumentor().instrument(tracer_provider=provider)` so ADK activity is emitted as OpenTelemetry spans, which the exporter sends to Langfuse.

**Call sites** (examples): `report_generation/agent.py`, `aml_investigation/agent.py`, `knowledge_qa/cli.py`, `summarization/agent.py`, `entity_extraction/agent.py`, and `entity_extraction/cli.py` when tracing is enabled.

**Related:** `set_up_langfuse_otlp_env_vars()` and `setup_langfuse_tracer()` in the same module configure OTLP env vars / a simpler span processor path; the primary path used for ADK is `init_tracing()`.

### 2. Langfuse Python SDK (HTTP API)

**Purpose:** Everything that is **not** pure OTLP span export: datasets, experiment runs, reading traces back, and attaching **scores**.

The `Langfuse` instance from `AsyncClientManager` talks to Langfuse’s public API (the SDK handles HTTP). Notable operations in this repo:

| Operation | Where | What goes to Langfuse |
|-----------|--------|------------------------|
| `create_dataset` / `get_dataset` | `upload_dataset_to_langfuse()` in `langfuse.py` | Dataset container |
| `create_dataset_item` | same | Per-row `input`, `expected_output`, `metadata`, deterministic `id` |
| `get_dataset` + `dataset.run_experiment(...)` | `evaluation/experiment.py` | Experiment execution; Langfuse records runs, links outputs/traces per its SDK behavior |
| `async_api.trace.get(trace_id)` | `evaluation/trace.py` | Fetches full trace for trace-level evaluators |
| `create_score(...)` | `evaluation/trace.py` (`_upload_trace_scores`), `langfuse.py` (`_report_score`, `report_usage_scores`), `report_generation/evaluation/online.py` | Scores linked to `trace_id` (and optional metadata / data types) |
| `flush()` | `AsyncClientManager.close()`, `flush_traces()`, various score reporters | Ensures buffered SDK traffic is sent |

**Trace evaluation scores:** After trace-level evaluators run, `_upload_trace_scores` maps each `langfuse.experiment.Evaluation` to `langfuse_client.create_score`, including numeric, boolean, and categorical string values.

**Online report-generation scoring:** `report_final_response_score()` in `report_generation/evaluation/online.py` uses `get_current_trace_id()` (must run inside an active Langfuse trace context), then `create_score` for “Valid Final Response” and `flush()`.

**Usage thresholds:** `report_usage_scores()` in `langfuse.py` loads a trace via `fetch_trace_with_wait`, derives metrics with `extract_trace_metrics`, and may post binary pass/fail scores (“Token Count”, “Latency”, “Cost”) via `create_score`.

## Dataset upload helpers

- **Generic uploader:** `upload_dataset_to_langfuse(dataset_path, dataset_name)` in `langfuse.py` — accepts `.json` (array) or `.jsonl`; each record needs `input` and `expected_output`.
- **CLI wrappers:** e.g. `implementations/report_generation/data/langfuse_upload.py`, `implementations/knowledge_qa/data/langfuse_upload.py` call into that function.

Upload ends with `await client_manager.close()` so the Langfuse client is flushed and torn down after the batch.

## Evaluation harness and Langfuse

- `run_experiment()` wraps Langfuse’s `dataset.run_experiment` after `get_dataset(dataset_name)`.
- `run_experiment_with_trace_evals()` adds a second pass: `run_trace_evaluations()` fetches each trace, runs trace evaluators, uploads scores with `_upload_trace_scores`, then `flush_traces()`.

Graders such as DeepSearchQA produce `langfuse.experiment.Evaluation` objects (e.g. `deepsearchqa_grader.py` `to_evaluations()`) that the harness or trace upload path turns into Langfuse scores.

## Other repo scripts

`implementations/shared/delete_langfuse_dataset.py` uses a **small direct REST client** (Basic auth with public/secret key) to delete datasets via Langfuse’s HTTP API — separate from the main `langfuse` package flows above.

## Summarization and entity extraction agents

Both agents use the same `init_tracing()` OTLP path described above, but tracing is **off by default** (`langfuse_tracing=False`). Opt in explicitly:

| Agent | How to enable |
|-------|---------------|
| **SummarizationAgent** | `SummarizationAgent(langfuse_tracing=True)` or via `SummarizationAgentManager(langfuse_tracing=True)` |
| **Entity extraction** | `run_entity_extraction(title, maintext, langfuse_tracing=True)` |
| **Entity extraction CLI** | `entity-extract run --traces ...` |
| **Workflow script** | `python scripts/run_workflow.py --traces` (applies to both agents and evaluation uploads) |

`create_entity_extraction_agent()` is a pure factory with no tracing side effects. If you build a custom `Runner` from it, call `init_tracing()` yourself before the first `run_async`.

## Quick mental model

1. **OTLP** → live **traces/spans** from instrumented agents (`init_tracing` + ADK).
2. **Langfuse SDK** → **datasets**, **experiments**, **trace reads**, and **scores** (`create_score`, `flush`), all authenticated with the same project keys and host.

If traces or scores look missing, check that keys/host match your Langfuse project and that `flush()` / `AsyncClientManager.close()` run after short-lived scripts.
