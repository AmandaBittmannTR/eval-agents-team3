# Summarization Agent Evaluation

This directory contains the evaluation setup for the Financial News Summarization Agent using LLM-as-a-judge methodology, following the same pattern as the Knowledge QA experiment.

## Overview

The summarization evaluation assesses the quality of AI-generated summaries across four key dimensions:

- **Accuracy** (0-1): Factual correctness, absence of hallucinations
- **Completeness** (0-1): Coverage of key information from the original article
- **Conciseness** (0-1): Appropriate brevity without unnecessary details
- **Clarity** (0-1): Readability, coherence, and professional language

## Files

- `evaluate.py` - Main evaluation script using Langfuse experiments
- `agent.py` - ADK discovery entrypoint for the summarization agent
- `../shared/langfuse_upload.py` - Shared script for uploading datasets to Langfuse

## Quick Start

### 1. Upload Dataset to Langfuse

```bash
# Upload dataset directly from CSV using the shared script
python implementations/shared/langfuse_upload.py \
    --dataset-name "Financial-News-Summarization" \
    --dataset-path data/transformed_data/2020_data.csv \
    --samples 20 \
    --evaluation-type summarization
```


### 3. Run Evaluation

```bash
# Run the evaluation experiment
python evaluate.py --dataset-name "Financial-News-Summarization"

# Or with custom settings
python evaluate.py \
    --dataset-name "Financial-News-Summarization" \
    --experiment-name "Summarization-v1" \
    --max-concurrency 2
```

## Evaluation Methodology

### LLM-as-a-Judge Approach

The evaluation uses a specialized LLM judge (similar to DeepSearchQA's approach) rather than the generic `create_llm_as_judge_evaluator`. Key features:

- **Custom Grader**: `aieng.agent_evals.summarization.summarization_grader`
- **Structured Output**: Uses Pydantic models for consistent evaluation format
- **Multi-dimensional Scoring**: Four quality dimensions plus overall assessment
- **Financial Domain Focus**: Prompts tailored for financial news content

### Evaluation Metrics

Each summary receives:

1. **Individual Scores** (0-1 scale):
   - Accuracy: Factual correctness
   - Completeness: Key information coverage  
   - Conciseness: Appropriate brevity
   - Clarity: Readability and coherence

2. **Aggregate Metrics**:
   - Average Score: Mean of the four dimensions
   - Overall Quality: Categorical rating (excellent/good/fair/poor)

3. **Langfuse Integration**:
   - All metrics logged as separate Evaluations
   - Rich metadata including reasoning chains
   - Experiment comparison and tracking

### Ground Truth Options

You have two options for ground truth summaries:

1. **Use existing descriptions** (default with shared script): Uses the `description` field from the CSV as ground truth
2. **Manual annotation** (if needed): Create custom reference summaries for specialized evaluation scenarios

## Dataset Format

The Langfuse dataset (created by the shared upload script) uses this format:

```json
{
  "input": {
    "title": "Article headline",
    "maintext": "Full article text...",
    "description": "Article description (optional)"
  },
  "expected_output": "Reference summary text (from description field)",
  "metadata": {
    "example_id": 0,
    "source": "financial_news",
    "evaluation_type": "summarization",
    "article_length": 1500,
    "has_description": true,
    "description_length": 150,
    "entity_count": 5,
    "company_count": 2,
    "summary_source": "description"
  }
}
```

## Configuration

### Environment Variables

Required environment variables (same as other evaluations):

```bash
# Langfuse configuration
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# OpenAI configuration (for LLM judge)
OPENAI_API_KEY=sk-...

# Google AI configuration (for summarization agent)
GOOGLE_API_KEY=...
```

### Optional Settings

```bash
# Enable trace-level groundedness evaluation
ENABLE_TRACE_GROUNDEDNESS=true
```

## Advanced Usage

### Custom Evaluation Criteria

To modify the evaluation criteria, edit the prompt in `aieng/agent_evals/summarization/summarization_grader.py`:

- Adjust scoring guidelines
- Add domain-specific requirements
- Modify quality thresholds

### Trace-Level Evaluation

Enable trace groundedness evaluation to check if summaries are supported by the original article:

```bash
ENABLE_TRACE_GROUNDEDNESS=true python evaluate.py
```

### Batch Processing

For large-scale evaluation:

```bash
# Increase concurrency (be mindful of rate limits)
python evaluate.py --max-concurrency 5

# Process larger datasets
python implementations/shared/langfuse_upload.py --dataset-name "Large-Dataset" --dataset-path data/transformed_data/2020_data.csv --samples 100 --evaluation-type summarization
```

## Integration with Evaluation Framework

This implementation follows the same patterns as the Knowledge QA evaluation:

- Uses `run_experiment()` from `aieng.agent_evals.evaluation`
- Compatible with trace-level evaluators
- Supports the same CLI options and configuration
- Integrates with the shared async client manager

## Troubleshooting

### Common Issues

1. **Missing Ground Truth**: If articles don't have expected_output, the evaluation will still run but comparisons may be limited.

2. **Rate Limiting**: Reduce `--max-concurrency` or add delays in data preparation.

3. **Large Articles**: Very long articles may hit context limits. Consider preprocessing to truncate or summarize.

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Langfuse dashboard for experiment results and trace details.

## Comparison with Knowledge QA

| Aspect | Knowledge QA | Summarization |
|--------|-------------|---------------|
| **Grader** | DeepSearchQA-specific | Summarization-specific |
| **Metrics** | Precision, Recall, F1, Outcome | Accuracy, Completeness, Conciseness, Clarity |
| **Evaluation** | Set-based matching | Multi-dimensional quality assessment |
| **Domain** | Research questions | Financial news |
| **Ground Truth** | Exact answer matching | Reference summary comparison |

Both use the same underlying evaluation harness and Langfuse integration patterns.