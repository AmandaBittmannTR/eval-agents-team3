# Basics: Agent Evaluation Fundamentals

Two introductory notebooks covering the foundations of agent evaluation.

- **`01_why_evals.ipynb`** — Conceptual overview: why agent evaluation is hard, the four quality dimensions (outcome, tool usage, reasoning, cost), and the three grader types (code-based, model-based, human). No code execution required.
- **`02_evaluation_harness.ipynb`** — Hands-on walkthrough of the shared evaluation harness: uploading datasets to Langfuse, writing task and evaluator functions, using `create_llm_as_judge_evaluator`, and running two-pass trace evaluations.

## Setup

1. Copy and configure the environment file:

```bash
cp .env.example .env
```

2. Set the required API keys:

```bash
GOOGLE_API_KEY="your-api-key"       # Used by google-adk agents
LANGFUSE_PUBLIC_KEY="pk-lf-..."     # Langfuse tracing
LANGFUSE_SECRET_KEY="sk-lf-..."
```

3. Install dependencies:

```bash
uv sync
```

## Running the Notebooks

Open in Jupyter or VS Code and run cells in order. The working directory is automatically corrected to the repo root in each notebook's setup cell.
