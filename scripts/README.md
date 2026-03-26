# Scripts

## Knowledge QA workflow (`run_workflow.py`)

End-to-end entry point for the evaluation pipeline: load news articles from a local CSV **or** a Langfuse dataset, run **entity extraction** and/or **summarization** agents (Google ADK / Gemini) in parallel, automatically evaluate results (P/R/F1, TF-IDF cosine + BERTScore, LLM-as-a-judge), and optionally push evaluation scores to Langfuse.

### Prerequisites

- **Python 3.12+** and a project environment with the workspace package installed (recommended: **uv**).
- A **`.env` file in the repository root** (same folder as `pyproject.toml`) with at least one API key the app understands. See [`.env.example`](../.env.example). Accepted names include `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` (Gemini key).

The script calls `load_dotenv()` and then syncs credentials into `GOOGLE_API_KEY` / `GEMINI_API_KEY` in the process environment so the Google GenAI SDK can authenticate. Run from the **repo root** so `.env` is found.

### Install / run environment

From the repository root:

```powershell
cd c:\path\to\eval-agents-team3
uv sync
```

### How to run

Always use the **repository root** as the working directory so paths like `data/transformed_data/...` resolve and `.env` loads correctly.

**Recommended (uses the workspace venv and `aieng-eval-agents`):**

```powershell
cd c:\path\to\eval-agents-team3
uv run python scripts/run_workflow.py --help
```

**Examples**

```powershell
# Defaults: local CSV, both agents, parallel pipelines
uv run python scripts/run_workflow.py

# Limit rows (useful for smoke tests)
uv run python scripts/run_workflow.py --sample-size 5

# Another year / file
uv run python scripts/run_workflow.py --data-file data/transformed_data/2018_data.csv --sample-size 10

# Load articles from a Langfuse dataset instead of CSV
uv run python scripts/run_workflow.py --dataset-name FinancialNews-2017 --sample-size 5

# Only entity extraction or only summarization
uv run python scripts/run_workflow.py --agents entity-extraction
uv run python scripts/run_workflow.py --agents summarization

# Run pipelines one after the other (default is parallel)
uv run python scripts/run_workflow.py --sequential

# Enable Langfuse tracing + push eval scores
uv run python scripts/run_workflow.py --traces

# Custom output directory
uv run python scripts/run_workflow.py --output-dir my_outputs
```

### CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | `data/transformed_data/2017_data.csv` | Input CSV path (relative to cwd or absolute). Mutually exclusive with `--dataset-name`. |
| `--dataset-name` | `FinancialNews-2017` | Load articles from a Langfuse dataset instead of a local CSV. Mutually exclusive with `--data-file`. |
| `--sample-size` | *(all rows)* | Maximum number of articles to process. |
| `--output-dir` | `outputs` | Directory for the JSON results file. |
| `--agents` | `entity-extraction summarization` | One or both of: `entity-extraction`, `summarization`. |
| `--sequential` | off | Run pipelines one after the other instead of concurrently (default is parallel). |
| `--traces` | off | Enable Langfuse tracing via OpenTelemetry for agents and push evaluation scores to Langfuse. |

### Input data format

The CSV must include these columns:

- `title`, `maintext` — agent input  
- `description` — reference text for summarization evaluation  
- `mentioned_companies`, `named_entities` — reference data for entity extraction evaluation  

`mentioned_companies` and `named_entities` are usually stringified Python literals in the CSV (lists), which the script parses with `ast.literal_eval`.

### Output

- **`{output-dir}/workflow_result.json`** — full run payload (per-article results, errors, timings).  
- **Console** — Rich table summary and evaluation results (P/R/F1, similarity, LLM-as-judge).

The `outputs/` directory is listed in `.gitignore` at the repo root.

### Troubleshooting

- **`Missing key inputs argument` / api_key** — Ensure a non-empty Gemini key is in `.env` under one of the supported variable names. Do not leave `GOOGLE_API_KEY=` or `GEMINI_API_KEY=` blank; remove the line or set a real value.  
- **Wrong cwd** — If `.env` or `data/...` is not found, `cd` to the repo root before running.  
- **Costs and rate limits** — Full files are large; use `--sample-size` first.

### Evaluation

After each agent completes, evaluations run automatically:

- **Entity extraction**: precision, recall, and F1 for companies, named entities, and words (from `entity_extraction_eval.py`).
- **Summarization similarity**: TF-IDF cosine similarity vs. reference description + BERTScore F1 vs. article (from `summary_similarity_eval.py`).
- **Summarization LLM-as-judge**: accuracy, completeness, conciseness, and clarity scored by an LLM grader (from the `summarization_grader` package).

Use `--traces` to push evaluation scores to Langfuse alongside agent traces.
