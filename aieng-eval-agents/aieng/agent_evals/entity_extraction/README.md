# Entity extraction CLI — reading the output

This document explains what each part of the terminal output from `cli.py` means and how to compare it to ground-truth rows when you use `--ground-truth`.

## How to run

From the repo root, use [uv](https://docs.astral.sh/uv/) so the command runs with the project’s Python and dependencies. Configure `GOOGLE_API_KEY` / ADK credentials as for the rest of `aieng-eval-agents`.

```bash
uv run python -m aieng.agent_evals.entity_extraction.cli run --file data/transformed_data/2017_data.csv --rows 2 --ground-truth
```

If the package exposes the `entity-extract` console script (see package config), the same flags work as:

```bash
uv run entity-extract run --file data/transformed_data/2017_data.csv --rows 2 --ground-truth
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--file` | Path to the transformed CSV (`title`, `maintext`, label columns). |
| `--rows N` | Process the first `N` data rows (after the header). |
| `--row-index i j …` | Process only those zero-based row indices (overrides the “first N rows” behaviour). |
| `--title` / `--maintext` | Run on a single inline article instead of a file. |
| `--langfuse-trace` | Enable Langfuse tracing via OpenTelemetry (off by default). |
| `--ground-truth` | After each article, print the CSV’s label columns for side-by-side comparison. |

The banner shows the package version and the **worker model** from `Configs` (what actually runs the LLM).

---

## Sections in the output

### `--- Article i/N ---`

Progress marker: you are on article index `i` out of `N` in this run.

### Article panel (`[i/N] Article: …`)

The **headline** from the CSV (or your `--title`), truncated in the panel if it is very long. The model always receives both `title` and `maintext`.

### Mentioned Companies

**What it is:** ticker symbols the model believes are **explicitly supported by the article** (e.g. symbols written in the text or clearly tied to a company mentioned there). The prompt asks for tickers only, not free-form company names.

**How to read it:**

- A comma-separated list of tickers, or **`none`** when the model returns an empty list.
- This is **not** the same as “every company name in the article”; it is specifically **tickers** the model extracted under those rules.

### Entity table (Named entities)

Each row is one **deduplicated** surface form the model chose as a named entity:

| Column | Meaning |
|--------|---------|
| **Entity** | Text **exactly as it appears** in the title or body (`word`). |
| **Group** | Coarse type: `ORG` (organisation), `PER` (person), `LOC` (location), `MISC` (other). |
| **Normalized** | Optional canonical hint from the text (often a **ticker** for a company when the model attaches one). Shown as `-` when `null` / absent. |

**Interpretation tips:**

- The same real-world company might appear once (e.g. “Apple” with normalized `AAPL`) even if the article repeats the name.
- Product names, colours, protocols, or vague labels are often `MISC` even if they look like brands.
- **`Normalized`** is “best effort” from the model; it is not a guarantee of exchange-grade ticker accuracy.

### Ground Truth (only with `--ground-truth`)

This panel is **not** produced by the LLM. It prints the raw **label columns** from the CSV row you just ran:

- **Companies:** whatever is stored in that column (often a string that looks like a Python list, e.g. `['AAPL']`). This came from your dataset pipeline, not from the CLI formatter.
- **Entities:** often a **long serialized structure** (list of dicts with spans, scores, `company_key`, etc.). The CLI **truncates** long text so the panel stays readable; open the CSV in an editor if you need the full JSON.

**How to use it:** treat ground truth as a **reference label** from your data pipeline. The LLM output follows a **different schema** (simpler entities, tickers-only “mentioned companies”), so mismatches are expected when:

- Labels use NER spans and confidence scores and the model does not.
- Tickers in ground truth reflect a different labelling policy than the model’s prompt.
- The article text or preprocessing differs slightly from what was labelled.

The value is **spot-checking**: does the model roughly align on key orgs/locations/people, and are ticker lists in the right ballpark for finance-heavy articles?

### Closing line

`Done — processed N article(s)` means the run finished successfully for all `N` rows in that batch (network and API errors would surface as tracebacks instead).

---

## Short example (mental model)

For a gadget article, you might see **Mentioned Companies:** `FIT, P, AAPL` while the model lists entities such as “Fitbit”, “Apple”, “Pandora” with **Normalized** tickers where it inferred them. The **Ground Truth** row might list only `['AAPL']` for companies: that is a **label** vs **model extraction** difference, not a bug in the CLI.

For an article where the model finds **no** tickers, **Mentioned Companies** shows `none`, but **Entities** can still list organisations and people (`ORG`, `PER`, etc.) because those columns answer different questions.
