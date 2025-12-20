"""
Test script to verify CKAN API discovery works for all datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.bronze.dataset_config import TORONTO_DATASETS, get_all_dataset_keys
from ingestion.bronze.csv_loader import discover_resource_id, setup_logging
from loguru import logger

setup_logging()

def test_ckan_discovery():
    """Test CKAN API discovery for all datasets."""
    logger.info("=" * 60)
    logger.info("Testing CKAN API Discovery")
    logger.info("=" * 60)
    
    all_keys = get_all_dataset_keys()
    logger.info(f"Testing {len(all_keys)} datasets...")
    
    results = {
        "success": [],
        "failed": [],
        "already_configured": [],
    }
    
    for dataset_key in all_keys:
        dataset_info = TORONTO_DATASETS[dataset_key]
        logger.info("")
        logger.info(f"Testing: {dataset_info['name']} ({dataset_key})")
        
        if dataset_info.get("resource_id"):
            logger.info(f"  ✓ Already configured with resource ID: {dataset_info['resource_id']}")
            results["already_configured"].append(dataset_key)
        else:
            logger.info(f"  Discovering resource ID...")
            discovered = discover_resource_id(dataset_key)
            
            if discovered:
                if discovered.startswith("http"):
                    logger.success(f"  ✓ Discovered direct URL: {discovered[:80]}...")
                else:
                    logger.success(f"  ✓ Discovered resource ID: {discovered}")
                results["success"].append(dataset_key)
            else:
                logger.error(f"  ✗ Failed to discover resource ID")
                results["failed"].append(dataset_key)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Discovery Summary")
    logger.info("=" * 60)
    logger.info(f"Already configured: {len(results['already_configured'])}")
    logger.info(f"Successfully discovered: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        logger.warning(f"\nFailed datasets:")
        for key in results["failed"]:
            logger.warning(f"  - {key}")
    
    return len(results["failed"]) == 0


if __name__ == "__main__":
    success = test_ckan_discovery()
    sys.exit(0 if success else 1)

