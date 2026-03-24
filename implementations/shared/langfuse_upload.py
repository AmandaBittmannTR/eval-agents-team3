"""Unified dataset upload for both entity extraction and summarization experiments.

This script uploads financial news data to Langfuse in a format that works for both
entity extraction and summarization evaluations.

Usage:
    # Upload for both entity extraction and summarization
    python langfuse_upload.py --dataset-path data/transformed_data/2017_data.csv --dataset-name "FinancialNews-2017"
    
    # Upload with specific evaluation focus
    python langfuse_upload.py --dataset-path data/transformed_data/2017_data.csv --dataset-name "FinancialNews-2017" --evaluation-type both
"""

import asyncio
import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_financial_news_data(file_path: str) -> list[dict[str, Any]]:
    """Load financial news data from CSV, JSON, or JSONL file."""
    data = []
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        # Load CSV format (recommended)
        logger.info("Loading CSV format...")
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                try:
                    # Parse JSON strings back to objects
                    processed_row = {
                        "title": row["title"],
                        "maintext": row["maintext"],
                        "description": row.get("description", ""),
                        "mentioned_companies": json.loads(row["mentioned_companies"]) if row["mentioned_companies"] else [],
                        "named_entities": json.loads(row["named_entities"]) if row["named_entities"] else []
                    }
                    data.append(processed_row)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid CSV row {row_num}: {e}")
        logger.info(f"Loaded {len(data)} records from CSV format")
    
    else:
        # Load JSON/JSONL format
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # Try to load as single JSON array first (original format)
            try:
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    data = json_data
                    logger.info(f"Loaded {len(data)} records from JSON array format")
                else:
                    data = [json_data]
                    logger.info("Loaded 1 record from single JSON object")
            except json.JSONDecodeError:
                # Fall back to JSONL format
                logger.info("JSON array format failed, trying JSONL format...")
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                logger.info(f"Loaded {len(data)} records from JSONL format")
    
    return data


async def upload_financial_news_to_langfuse(
    dataset_name: str,
    dataset_path: str | None = None,
    samples: int | None = None,
    evaluation_type: Literal["entity_extraction", "summarization", "both"] = "both",
) -> None:
    """Upload financial news data to Langfuse for evaluation.

    Parameters
    ----------
    dataset_name : str
        Name for the dataset in Langfuse.
    dataset_path : str, optional
        Path to existing CSV/JSON/JSONL file with financial news data.
    samples : int, optional
        Number of samples to create (if dataset_path not provided).
    evaluation_type : str, optional
        Type of evaluation to optimize for: "entity_extraction", "summarization", or "both".
    """
    if dataset_path:
        # Load from existing file
        examples = load_financial_news_data(dataset_path)
        if samples:
            examples = examples[:samples]
            logger.info(f"Limited to first {samples} examples")
    else:
        # Create sample data for testing
        if not samples:
            samples = 10
        
        logger.info(f"Creating {samples} sample financial news examples")
        examples = []
        for i in range(samples):
            examples.append({
                "title": f"Sample Financial News {i+1}",
                "maintext": f"This is sample financial news content {i+1}. Apple Inc. (AAPL) reported strong quarterly earnings, beating analyst expectations. The technology giant saw revenue growth of 15% year-over-year, driven by strong iPhone and services performance.",
                "description": f"Apple Inc. exceeded quarterly expectations with 15% revenue growth driven by iPhone and services.",
                "named_entities": [
                    {"entity_group": "ORG", "word": "Apple Inc.", "normalized": "AAPL"},
                    {"entity_group": "MISC", "word": "iPhone", "normalized": None}
                ],
                "mentioned_companies": ["AAPL"]
            })

    if not examples:
        logger.error("No examples found")
        return

    # Convert examples to Langfuse format
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix=f"financial_news_{dataset_name}_",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        logger.info(f"Writing {len(examples)} examples to temporary file...")

        for i, example in enumerate(examples):
            # Shared input format (works for both entity extraction and summarization)
            input_data = {
                "title": example["title"],
                "maintext": example["maintext"]
            }
            
            # Create expected outputs for both evaluation types
            if evaluation_type in ["entity_extraction", "both"]:
                entity_expected_output = {
                    "named_entities": example.get("named_entities", []),
                    "mentioned_companies": example.get("mentioned_companies", [])
                }
            
            if evaluation_type in ["summarization", "both"]:
                # Use description as expected summary, fallback to truncated maintext
                summary_expected_output = example.get("description", "")
                if not summary_expected_output and example.get("maintext"):
                    # Create a simple extractive summary from first sentence
                    sentences = example["maintext"].split('. ')
                    summary_expected_output = sentences[0] + '.' if sentences else ""
            
            # Choose expected output based on evaluation type
            if evaluation_type == "entity_extraction":
                expected_output = json.dumps(entity_expected_output)
            elif evaluation_type == "summarization":
                expected_output = summary_expected_output
            else:  # both - default to entity extraction format, summarization will ignore
                expected_output = json.dumps(entity_expected_output)
            
            # Create metadata
            metadata = {
                "example_id": i,
                "source": "financial_news",
                "evaluation_type": evaluation_type,
                "article_length": len(example.get("maintext", "")),
                "has_description": bool(example.get("description", "").strip()),
                "entity_count": len(example.get("named_entities", [])),
                "company_count": len(example.get("mentioned_companies", [])),
                # Store both expected outputs in metadata for flexibility
                "entity_expected_output": json.dumps(entity_expected_output) if evaluation_type in ["entity_extraction", "both"] else None,
                "summary_expected_output": summary_expected_output if evaluation_type in ["summarization", "both"] else None
            }
            
            record = {
                "input": json.dumps(input_data),
                "expected_output": expected_output,
                "metadata": metadata,
            }
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        # Upload to Langfuse
        await upload_dataset_to_langfuse(
            dataset_path=str(temp_path),
            dataset_name=dataset_name,
        )
        logger.info(f"Successfully uploaded dataset '{dataset_name}' for {evaluation_type} evaluation(s)")
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Removed temporary file: {temp_path}")


@click.command()
@click.option(
    "--dataset-name",
    required=True,
    help="Name for the dataset in Langfuse.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to CSV/JSON/JSONL file with financial news data.",
)
@click.option(
    "--samples",
    type=int,
    help="Number of samples to upload (creates sample data if no dataset-path provided).",
)
@click.option(
    "--evaluation-type",
    type=click.Choice(["entity_extraction", "summarization", "both"]),
    default="both",
    help="Type of evaluation to optimize for (default: both).",
)
def cli(dataset_name: str, dataset_path: str | None, samples: int | None, evaluation_type: str) -> None:
    """Upload financial news data to Langfuse for entity extraction and/or summarization evaluation."""
    asyncio.run(upload_financial_news_to_langfuse(dataset_name, dataset_path, samples, evaluation_type))


if __name__ == "__main__":
    cli()