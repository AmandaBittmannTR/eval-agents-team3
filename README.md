# Agent Bootcamp

----------------------------------------------------------------------------------------

This is a collection of reference implementations for Vector Institute's **Agentic AI Evaluation Bootcamp**.

## Reference Implementations

This repository includes several modules, each showcasing a different aspect of agent-based RAG systems:

**3. Evals: Automated Evaluation Pipelines**
  Contains scripts and utilities for evaluating agent performance using LLM-as-a-judge and synthetic data generation. Includes tools for uploading datasets, running evaluations, and integrating with [Langfuse](https://langfuse.com/) for traceability.

- **[3.1 LLM-as-a-Judge](src/3_evals/1_llm_judge/README.md)**
  Automated evaluation pipelines using LLM-as-a-judge with Langfuse integration.

- **[3.2 Evaluation on Synthetic Dataset](src/3_evals/2_synthetic_data/README.md)**
  Showcases the generation of synthetic evaluation data for testing agents.


## Getting Started

Set your API keys in `.env`. Use `.env.example` as a template.

```bash
cp -v .env.example .env
```

Run integration tests to validate that your API keys are set up correctly.

```bash
uv run --env-file .env pytest -sv tests/tool_tests/test_integration.py
```

## Reference Implementations

For "Gradio App" reference implementations, running the script would print out a "public URL" ending in `gradio.live` (might take a few seconds to appear.) To access the gradio app with the full streaming capabilities, copy and paste this `gradio.live` URL into a new browser tab.

For all reference implementations, to exit, press "Ctrl/Control-C" and wait up to ten seconds. If you are a Mac user, you should use "Control-C" and not "Command-C". Please note that by default, the gradio web app reloads automatically as you edit the Python script. There is no need to manually stop and restart the program each time you make some code changes.

You might see warning messages like the following:

```json
ERROR:openai.agents:[non-fatal] Tracing client error 401: {
  "error": {
    "message": "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

These warnings can be safely ignored, as they are the result of a bug in the upstream libraries. Your agent traces will be uploaded to LangFuse as configured.

### 3. Evals

Synthetic data.

```bash
uv run --env-file .env \
-m src.3_evals.2_synthetic_data.synthesize_data \
--source_dataset hf://vector-institute/hotpotqa@d997ecf:train \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--limit 18
```

Quantify embedding diversity of synthetic data

```bash
# Baseline: "Real" dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset \
--run_name cosine_similarity_bge_m3

# Synthetic dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name cosine_similarity_bge_m3
```

Visualize embedding diversity of synthetic data

```bash
uv run \
--env-file .env \
gradio src/3_evals/2_synthetic_data/gradio_visualize_diversity.py
```

Run LLM-as-a-judge Evaluation on synthetic data

```bash
uv run \
--env-file .env \
-m src.3_evals.1_llm_judge.run_eval \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name enwiki_weaviate \
--limit 18
```

## Requirements

- Python 3.12+
- API keys as configured in `.env`.

### Tidbit

If you're curious about what "uv" stands for, it appears to have been more or
less chosen [randomly](https://github.com/astral-sh/uv/issues/1349#issuecomment-1986451785).
