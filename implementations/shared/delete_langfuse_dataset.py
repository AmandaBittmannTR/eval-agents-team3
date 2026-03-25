"""Delete Langfuse dataset using REST API.

This script uses Langfuse's REST API to delete a dataset and all its items.

Usage:
    python delete_dataset_api.py --dataset-name "FinancialNews-2017"
"""

import logging
import os
from typing import Optional

import click
import requests
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LangfuseAPI:
    """Simple Langfuse REST API client for dataset management."""
    
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
        
        if not self.public_key or not self.secret_key:
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in environment")
        
        self.base_url = f"{self.host}/api/public"
        self.auth = (self.public_key, self.secret_key)
    
    def list_datasets(self) -> list[dict]:
        """List all datasets."""
        try:
            response = requests.get(f"{self.base_url}/datasets", auth=self.auth)
            if response.status_code == 404:
                logger.info("No datasets endpoint found or no datasets exist")
                return []
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("data", result.get("datasets", []))
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def get_dataset_items(self, dataset_name: str) -> list[dict]:
        """Get all items in a dataset with pagination support."""
        try:
            all_items = []
            page = 1
            limit = 50  # Default page size
            
            # Try different possible endpoints
            endpoints = [
                f"{self.base_url}/datasets/{dataset_name}/items",
                f"{self.base_url}/dataset-items"
            ]
            
            for endpoint in endpoints:
                try:
                    while True:
                        # Set up parameters for pagination
                        if "dataset-items" in endpoint:
                            params = {
                                "datasetName": dataset_name,
                                "page": page,
                                "limit": limit
                            }
                        else:
                            params = {
                                "page": page,
                                "limit": limit
                            }
                        
                        response = requests.get(endpoint, auth=self.auth, params=params)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if isinstance(result, list):
                                items = result
                            elif isinstance(result, dict):
                                items = result.get("data", result.get("items", []))
                                # Check if there's pagination info
                                total_count = result.get("totalCount", result.get("total", 0))
                                if total_count and page == 1:
                                    logger.info(f"Dataset has {total_count} total items, fetching all pages...")
                            else:
                                items = []
                            
                            # Filter by dataset name if needed
                            if "dataset-items" in endpoint and items:
                                items = [item for item in items if item.get("datasetName") == dataset_name]
                            
                            if not items:
                                # No more items, we're done
                                break
                                
                            all_items.extend(items)
                            logger.debug(f"Fetched page {page}: {len(items)} items (total so far: {len(all_items)})")
                            
                            # Check if we got fewer items than the limit (last page)
                            if len(items) < limit:
                                break
                                
                            page += 1
                            
                        elif response.status_code == 404:
                            break  # Try next endpoint
                        else:
                            logger.warning(f"Endpoint {endpoint} page {page} returned {response.status_code}")
                            break
                    
                    # If we got items from this endpoint, return them
                    if all_items:
                        logger.info(f"Successfully fetched {len(all_items)} items from {endpoint}")
                        return all_items
                        
                except requests.RequestException as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            logger.error(f"All endpoints failed for dataset '{dataset_name}'")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get dataset items: {e}")
            return []
    
    def delete_dataset_item(self, item_id: str) -> bool:
        """Delete a single dataset item."""
        try:
            response = requests.delete(
                f"{self.base_url}/dataset-items/{item_id}",
                auth=self.auth
            )
            return response.status_code in [200, 204, 404]  # 404 means already deleted
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}")
            return False


def delete_dataset(dataset_name: str, confirm: bool = False) -> bool:
    """Delete a dataset and all its items.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to delete.
    confirm : bool
        If True, skip confirmation prompt.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        api = LangfuseAPI()
        logger.info(f"Connected to Langfuse at {api.host}")
        
        # List datasets to verify it exists
        logger.info("Fetching available datasets...")
        datasets = api.list_datasets()
        dataset_names = [d.get("name") for d in datasets if d.get("name")]
        
        logger.info(f"Found {len(datasets)} datasets: {dataset_names}")
        
        if dataset_name not in dataset_names:
            logger.error(f"Dataset '{dataset_name}' not found")
            if dataset_names:
                logger.info(f"Available datasets: {dataset_names}")
            else:
                logger.info("No datasets found. The dataset may have been already deleted or never existed.")
            return False
        
        # Get dataset items
        logger.info(f"Fetching all items from dataset '{dataset_name}'...")
        items = api.get_dataset_items(dataset_name)
        logger.info(f"Found dataset '{dataset_name}' with {len(items)} items")
        
        if not items:
            logger.info(f"Dataset '{dataset_name}' is already empty")
            return True
        
        # Confirmation
        if not confirm:
            response = input(f"Delete all {len(items)} items from '{dataset_name}'? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Deletion cancelled")
                return False
        
        # Delete all items
        logger.info(f"Deleting {len(items)} items...")
        deleted_count = 0
        failed_count = 0
        
        for i, item in enumerate(items, 1):
            item_id = item.get("id")
            if item_id:
                if api.delete_dataset_item(item_id):
                    deleted_count += 1
                else:
                    failed_count += 1
                    logger.debug(f"Failed to delete item {item_id}")
            else:
                failed_count += 1
                logger.debug(f"Item {i} has no ID")
            
            # Progress indicator every 50 items or at the end
            if i % 50 == 0 or i == len(items):
                logger.info(f"Progress: {i}/{len(items)} processed, {deleted_count} deleted, {failed_count} failed")
        
        logger.info(f"Final result: {deleted_count} deleted, {failed_count} failed out of {len(items)} total items")
        
        if deleted_count == len(items):
            logger.info(f"✅ Dataset '{dataset_name}' has been cleared")
            return True
        else:
            logger.warning(f"⚠️ Only {deleted_count}/{len(items)} items were deleted")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        return False


def list_datasets() -> None:
    """List all available datasets."""
    try:
        api = LangfuseAPI()
        datasets = api.list_datasets()
        
        if not datasets:
            logger.info("No datasets found")
            return
        
        logger.info(f"Found {len(datasets)} datasets:")
        for dataset in datasets:
            name = dataset.get("name", "Unknown")
            created_at = dataset.get("createdAt", "Unknown")
            logger.info(f"  - {name} (created: {created_at})")
            
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")


@click.command()
@click.option(
    "--dataset-name",
    default="FinancialNews-2017",
    help="Name of the dataset to delete.",
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt.",
)
@click.option(
    "--list",
    "list_datasets_flag",
    is_flag=True,
    help="List all available datasets.",
)
def cli(dataset_name: str, confirm: bool, list_datasets_flag: bool) -> None:
    """Delete a Langfuse dataset by removing all its items."""
    
    if list_datasets_flag:
        list_datasets()
        return
    
    logger.info(f"Attempting to delete dataset: {dataset_name}")
    success = delete_dataset(dataset_name, confirm)
    
    if success:
        logger.info("🎉 Dataset deletion completed!")
        logger.info("You can now upload fresh data without duplicates.")
    else:
        logger.error("❌ Dataset deletion failed")


if __name__ == "__main__":
    cli()