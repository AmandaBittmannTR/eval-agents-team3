# Scripts

## Knowledge QA workflow (`run_workflow.py`)

End-to-end entry point for the evaluation pipeline: load news articles from a CSV, run **entity extraction** and/or **summarization** agents (Google ADK / Gemini), write aggregated JSON results, and print placeholder evaluation summaries to the console.

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
# Defaults: data/transformed_data/2017_data.csv, both agents, sequential pipelines
uv run python scripts/run_workflow.py

# Limit rows (useful for smoke tests)
uv run python scripts/run_workflow.py --sample-size 5

# Another year / file
uv run python scripts/run_workflow.py --data-file data/transformed_data/2018_data.csv --sample-size 10

# Only entity extraction or only summarization
uv run python scripts/run_workflow.py --agents entity-extraction
uv run python scripts/run_workflow.py --agents summarization

# Run both pipelines at the same time (higher load; optional)
uv run python scripts/run_workflow.py --parallel

# Custom output directory
uv run python scripts/run_workflow.py --output-dir my_outputs
```

### CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | `data/transformed_data/2017_data.csv` | Input CSV path (relative to cwd or absolute). |
| `--sample-size` | *(all rows)* | Maximum number of articles to process. |
| `--output-dir` | `outputs` | Directory for the JSON results file. |
| `--agents` | `entity-extraction summarization` | One or both of: `entity-extraction`, `summarization`. |
| `--parallel` | off | Run the two pipelines concurrently instead of one after the other. |
| `--langfuse-trace` | off | Enable Langfuse tracing via OpenTelemetry for both agents. |

### Input data format

The CSV must include these columns:

- `title`, `maintext` — agent input  
- `description` — reference text for future summarization evaluation  
- `mentioned_companies`, `named_entities` — reference data for future entity evaluation  

`mentioned_companies` and `named_entities` are usually stringified Python literals in the CSV (lists), which the script parses with `ast.literal_eval`.

### Output

- **`{output-dir}/workflow_result.json`** — full run payload (per-article results, errors, timings).  
- **Console** — Rich table summary and placeholder evaluation messages.

The `outputs/` directory is listed in `.gitignore` at the repo root.

### Troubleshooting

- **`Missing key inputs argument` / api_key** — Ensure a non-empty Gemini key is in `.env` under one of the supported variable names. Do not leave `GOOGLE_API_KEY=` or `GEMINI_API_KEY=` blank; remove the line or set a real value.  
- **Wrong cwd** — If `.env` or `data/...` is not found, `cd` to the repo root before running.  
- **Costs and rate limits** — Full files are large; use `--sample-size` first.

### What is not implemented yet

Evaluation metrics (code-based entity scores, LLM-as-a-judge for summaries) are placeholders; the script is focused on **running agents and capturing outputs**. Langfuse tracing is available via `--langfuse-trace`.
