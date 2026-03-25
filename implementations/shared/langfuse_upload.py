"""Unified dataset upload for both entity extraction and summarization experiments.

This script uploads financial news data to Langfuse in a format that works for both
entity extraction and summarization evaluations.

Usage:
    # Upload for both entity extraction and summarization
    python implementations/shared/langfuse_upload.py --dataset-path data/transformed_data/2017_data.csv --dataset-name "FinancialNews-2017"
    
    # Upload with specific evaluation focus
    python implementations/shared/langfuse_upload.py --dataset-path data/transformed_data/2017_data.csv --dataset-name "FinancialNews-2017" --evaluation-type both
"""

import asyncio
import csv
import json
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Literal

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading data from different file formats."""
    
    @staticmethod
    def safe_json_parse(json_str: str, default=None) -> Any:
        """Safely parse JSON strings or Python literals."""
        if default is None:
            default = []
        if not json_str or json_str.strip() == "":
            return default
        
        # Try JSON parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fall back to Python literal_eval for single quotes
        try:
            import ast
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            logger.debug(f"Could not parse: {json_str[:50]}...")
            return default
    
    @staticmethod
    def load_csv(file_path: str) -> list[dict[str, Any]]:
        """Load data from CSV file."""
        logger.info("Loading CSV format...")
        data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    processed_row = {
                        "title": row.get("title", "").strip(),
                        "maintext": row.get("maintext", "").strip(),
                        "description": row.get("description", "").strip(),
                        "mentioned_companies": DataLoader.safe_json_parse(row.get("mentioned_companies", ""), []),
                        "named_entities": DataLoader.safe_json_parse(row.get("named_entities", ""), [])
                    }
                    
                    # Skip rows with empty title or maintext
                    if not processed_row["title"] or not processed_row["maintext"]:
                        logger.debug(f"Skipping row {row_num}: empty title or maintext")
                        continue
                    
                    data.append(processed_row)
                except Exception as e:
                    logger.warning(f"Skipping invalid CSV row {row_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} valid records from CSV format")
        return data
    
    @staticmethod
    def load_json(file_path: str) -> list[dict[str, Any]]:
        """Load data from JSON or JSONL file."""
        data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # Try JSON array format first
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
    
    @classmethod
    def load_data(cls, file_path: str) -> list[dict[str, Any]]:
        """Load data from file, auto-detecting format."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return cls.load_csv(file_path)
        else:
            return cls.load_json(file_path)

class DataTransformer:
    """Transforms data for different evaluation types."""
    
    @staticmethod
    def create_input_data(example: dict, evaluation_type: str) -> dict:
        """Create input data for agents."""
        input_data = {
            "title": example["title"],
            "maintext": example["maintext"]
        }
        
        # Add description for summarization context
        if evaluation_type in ["summarization", "both"] and example.get("description"):
            input_data["description"] = example["description"]
        
        return input_data
    
    @staticmethod
    def create_entity_expected_output(example: dict) -> dict:
        """Convert ground truth entities to agent format."""
        agent_format_entities = []
        
        for entity in example.get("named_entities", []):
            entity_group = entity.get("entity_group", "MISC")
            
            # Map entity groups to agent types
            if entity_group == "ORG":
                entity_type = "B"  # Business
            elif entity_group == "PER":
                entity_type = "P"  # Person
            else:
                entity_type = "M"  # Miscellaneous
            
            agent_entity = {
                "type": entity_type,
                "name": entity.get("word", ""),
                "ticker": entity.get("normalized") if entity.get("normalized") else None
            }
            agent_format_entities.append(agent_entity)
        
        return {"entities": agent_format_entities}
    
    @staticmethod
    def create_summary_expected_output(example: dict, row_index: int) -> str:
        """Create expected summary output."""
        summary = example.get("description", "").strip()
        
        # Fallback to first sentence if no description
        if not summary and example.get("maintext"):
            sentences = example["maintext"].split('. ')
            summary = sentences[0] + '.' if sentences else ""
        
        # Final fallback
        if not summary:
            summary = "No summary available"
            logger.warning(f"Row {row_index}: No description or maintext available for summarization")
        
        return summary
    
    @staticmethod
    def create_metadata(example: dict, row_index: int, evaluation_type: str) -> dict:
        """Create metadata for the record."""
        metadata = {
            "example_id": row_index,
            "source": "financial_news",
            "evaluation_type": evaluation_type,
            "article_length": len(example.get("maintext", "")),
            "has_description": bool(example.get("description", "").strip()),
            "description_length": len(example.get("description", "")),
            "entity_count": len(example.get("named_entities", [])),
            "company_count": len(example.get("mentioned_companies", []))
        }
        
        # Add evaluation-specific metadata
        if evaluation_type in ["summarization", "both"]:
            metadata["summary_source"] = "description" if example.get("description", "").strip() else "generated"
        
        if evaluation_type in ["entity_extraction", "both"]:
            ground_truth_format = {
                "named_entities": example.get("named_entities", []),
                "mentioned_companies": example.get("mentioned_companies", [])
            }
            metadata["ground_truth_format"] = json.dumps(ground_truth_format)
        
        return metadata


class LangfuseUploader:
    """Handles the upload process to Langfuse."""
    
    @staticmethod
    def create_deterministic_id(dataset_name: str, row_index: int, input_data: dict) -> str:
        """Create a deterministic ID to prevent duplicates."""
        input_hash = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()[:8]
        return f"{dataset_name}-{row_index}-{input_hash}"
    
    @staticmethod
    async def upload_to_langfuse(
        examples: list[dict], 
        dataset_name: str, 
        evaluation_type: str
    ) -> None:
        """Upload examples to Langfuse."""
        if not examples:
            logger.error("No examples to upload")
            return
        
        # Create temporary JSONL file
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
                # Create input data
                input_data = DataTransformer.create_input_data(example, evaluation_type)
                
                # Create expected output based on evaluation type
                if evaluation_type == "entity_extraction":
                    expected_output = json.dumps(DataTransformer.create_entity_expected_output(example))
                elif evaluation_type == "summarization":
                    expected_output = DataTransformer.create_summary_expected_output(example, i)
                else:  # both - include both entity extraction and summary
                    entity_output = DataTransformer.create_entity_expected_output(example)
                    summary_output = DataTransformer.create_summary_expected_output(example, i)
                    combined_output = {
                        "entities": entity_output["entities"],
                        "summary": summary_output
                    }
                    expected_output = json.dumps(combined_output)
                
                # Create metadata
                metadata = DataTransformer.create_metadata(example, i, evaluation_type)
                
                # Create record with deterministic ID
                record = {
                    "id": LangfuseUploader.create_deterministic_id(dataset_name, i, input_data),
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


async def upload_financial_news_to_langfuse(
    dataset_name: str,
    dataset_path: str | None = None,
    samples: int | None = None,
    evaluation_type: Literal["entity_extraction", "summarization", "both"] = "both",
) -> None:
    """Main upload function."""
    
    # Load or generate data
    if dataset_path:
        examples = DataLoader.load_data(dataset_path)
        if samples:
            examples = examples[:samples]
            logger.info(f"Limited to first {samples} examples")
    
    # Upload to Langfuse
    await LangfuseUploader.upload_to_langfuse(examples, dataset_name, evaluation_type)


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