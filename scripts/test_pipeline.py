"""
Test Pipeline: Process small dataset end-to-end for validation.

This script runs a minimal end-to-end test of the data pipeline:
1. Downloads a small sample (first N rows) from one dataset
2. Processes through Bronze → Silver → Gold layers
3. Provides detailed error reporting and progress tracking

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --rows 500 --dataset major-crime-indicators
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Configure logging for test pipeline."""
    # Fix Unicode encoding for Windows PowerShell
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            # Python < 3.7 or reconfigure not available
            pass
    
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )


def download_test_data(dataset_key: str = "major-crime-indicators", max_rows: int = 1000) -> Path:
    """
    Download a small sample of data for testing.
    
    Args:
        dataset_key: Dataset to download (default: major-crime-indicators)
        max_rows: Maximum number of rows to download (default: 1000)
        
    Returns:
        Path to downloaded test CSV file
    """
    logger.info("=" * 80)
    logger.info("TEST PIPELINE: BRONZE LAYER - Downloading Test Data")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_key}")
    logger.info(f"Max rows: {max_rows:,}")
    logger.info("")
    
    try:
        from ingestion.bronze.csv_loader import (
            discover_resource_id,
            download_csv,
            get_data_directory,
            TORONTO_DATASETS,
        )
        from ingestion.bronze.dataset_config import get_dataset_url
        
        # Get dataset info
        if dataset_key not in TORONTO_DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            logger.info(f"Available datasets: {', '.join(TORONTO_DATASETS.keys())}")
            sys.exit(1)
        
        dataset_info = TORONTO_DATASETS[dataset_key]
        logger.info(f"Dataset name: {dataset_info['name']}")
        
        # Get resource ID
        resource_id = dataset_info.get("resource_id")
        direct_url = dataset_info.get("direct_url")
        is_datastore = dataset_info.get("datastore_active", False)
        
        # Discover if needed
        if not resource_id and not direct_url:
            logger.info("Discovering resource ID...")
            discovered = discover_resource_id(dataset_key)
            if discovered:
                if discovered.startswith("http"):
                    direct_url = discovered
                else:
                    resource_id = discovered
        
        # Get download URL
        if direct_url:
            url = direct_url
        elif resource_id:
            if is_datastore:
                url = f"https://ckan0.cf.opendata.inter.prod-toronto.ca/datastore/dump/{resource_id}"
            else:
                url = get_dataset_url(dataset_key, resource_id)
        else:
            logger.error(f"Could not determine download URL for {dataset_key}")
            sys.exit(1)
        
        logger.info(f"Download URL: {url[:80]}...")
        
        # Download to temporary location first
        data_dir = get_data_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = data_dir / f"test_{dataset_key}_{timestamp}_full.csv"
        
        logger.info("Downloading full dataset...")
        if not download_csv(url, temp_path, is_datastore=is_datastore, resource_id=resource_id):
            logger.error("Failed to download dataset")
            sys.exit(1)
        
        logger.info(f"Downloaded to: {temp_path}")
        
        # Load and limit rows
        logger.info(f"Loading CSV and limiting to {max_rows:,} rows...")
        try:
            # Try to detect if it's an Excel file
            with open(temp_path, 'rb') as f:
                magic_bytes = f.read(4)
                is_excel = magic_bytes == b'PK\x03\x04'
            
            if is_excel:
                logger.info("Detected Excel file, reading with pandas...")
                df = pd.read_excel(temp_path, engine='openpyxl', nrows=max_rows)
            else:
                # Try to detect encoding
                try:
                    import chardet
                    with open(temp_path, 'rb') as f:
                        raw_data = f.read(10000)
                        result = chardet.detect(raw_data)
                        encoding = result['encoding'] or 'utf-8'
                except:
                    encoding = 'utf-8'
                
                # Read CSV with row limit
                df = pd.read_csv(
                    temp_path,
                    encoding=encoding,
                    nrows=max_rows,
                    low_memory=False,
                    on_bad_lines='skip',
                )
            
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Save limited dataset
            test_path = data_dir / f"test_{dataset_key}_{timestamp}_limited.csv"
            if is_excel:
                df.to_csv(test_path, index=False)
            else:
                df.to_csv(test_path, index=False, encoding='utf-8')
            
            logger.success(f"✅ Created test dataset: {test_path.name} ({len(df):,} rows)")
            
            # Clean up temp file
            try:
                temp_path.unlink()
                logger.info("Cleaned up temporary file")
            except:
                pass
            
            return test_path
            
        except Exception as e:
            logger.error(f"Error loading/limiting CSV: {e}")
            logger.exception(e)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in Bronze layer: {e}")
        logger.exception(e)
        sys.exit(1)


def process_bronze_test(csv_path: Path, dataset_key: str) -> bool:
    """
    Process test CSV through Bronze layer validation.
    
    Args:
        csv_path: Path to test CSV file
        dataset_key: Dataset key
        
    Returns:
        True if successful
    """
    logger.info("")
    logger.info("Processing through Bronze layer validation...")
    
    try:
        from ingestion.bronze.csv_loader import process_single_file
        
        result = process_single_file(csv_path, dataset_type=dataset_key, engine=None)
        
        if result:
            logger.success("✅ Bronze layer validation complete")
            return True
        else:
            logger.error("❌ Bronze layer validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bronze layer error: {e}")
        logger.exception(e)
        return False


def process_silver_test() -> bool:
    """
    Process test data through Silver layer.
    
    Returns:
        True if successful
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST PIPELINE: SILVER LAYER - Data Cleaning")
    logger.info("=" * 80)
    
    try:
        from ingestion.silver.cleaner import main as silver_main
        
        # Process only test files (files starting with "test_")
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        bronze_dir = project_root / "data" / "raw"
        
        test_files = list(bronze_dir.glob("test_*.csv"))
        if not test_files:
            logger.error("No test files found in Bronze layer")
            return False
        
        logger.info(f"Found {len(test_files)} test file(s) to process")
        
        # Process each test file
        for test_file in test_files:
            logger.info(f"Processing: {test_file.name}")
            try:
                # Call silver_main with input_file (process_all=False means use input_file)
                silver_main(input_file=str(test_file), process_all=False)
                logger.success(f"✅ Processed: {test_file.name}")
            except SystemExit as e:
                # SystemExit(0) means success
                if e.code == 0:
                    logger.success(f"✅ Processed: {test_file.name}")
                else:
                    logger.error(f"❌ Failed to process {test_file.name}")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to process {test_file.name}: {e}")
                logger.exception(e)
                return False
        
        logger.success("✅ Silver layer complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Silver layer error: {e}")
        logger.exception(e)
        return False


def process_gold_test() -> bool:
    """
    Process test data through Gold layer.
    
    Returns:
        True if successful
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST PIPELINE: GOLD LAYER - H3 Mapping & Database Loading")
    logger.info("=" * 80)
    
    try:
        from transformations.gold.h3_mapper import main as gold_main
        
        # Process only test files (files from test Silver files)
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        silver_dir = project_root / "data" / "silver"
        
        # Find Silver files that correspond to test files
        test_silver_files = list(silver_dir.glob("cleaned_test_*.csv"))
        if not test_silver_files:
            logger.error("No test Silver files found")
            return False
        
        logger.info(f"Found {len(test_silver_files)} test Silver file(s) to process")
        
        # Process with smaller batch size for testing
        for silver_file in test_silver_files:
            logger.info(f"Processing: {silver_file.name}")
            try:
                gold_main(
                    input_file=str(silver_file),
                    batch_size=500,  # Smaller batch size for testing
                    skip_existing=False,  # Don't skip for testing
                    verify=True,
                    process_all=False,
                    parallel=False,  # Sequential for testing
                )
                logger.success(f"✅ Processed: {silver_file.name}")
            except SystemExit as e:
                # SystemExit(0) means success
                if e.code == 0:
                    logger.success(f"✅ Processed: {silver_file.name}")
                else:
                    logger.error(f"❌ Failed to process {silver_file.name}")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to process {silver_file.name}: {e}")
                logger.exception(e)
                return False
        
        logger.success("✅ Gold layer complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gold layer error: {e}")
        logger.exception(e)
        return False


def cleanup_old_test_files(max_age_hours: int = 24) -> int:
    """
    Clean up old test files to prevent disk space issues.
    
    Args:
        max_age_hours: Delete test files older than this many hours (default: 24)
        
    Returns:
        Number of files deleted
    """
    project_root = Path(__file__).parent.parent
    bronze_dir = project_root / "data" / "raw"
    silver_dir = project_root / "data" / "silver"
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    deleted_count = 0
    
    # Clean up Bronze test files
    for test_file in bronze_dir.glob("test_*.csv"):
        try:
            file_time = datetime.fromtimestamp(test_file.stat().st_mtime)
            if file_time < cutoff_time:
                test_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old test file: {test_file.name}")
        except Exception as e:
            logger.warning(f"Could not delete {test_file.name}: {e}")
    
    # Clean up Silver test files
    for test_file in silver_dir.glob("cleaned_test_*.csv"):
        try:
            file_time = datetime.fromtimestamp(test_file.stat().st_mtime)
            if file_time < cutoff_time:
                test_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old test file: {test_file.name}")
        except Exception as e:
            logger.warning(f"Could not delete {test_file.name}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old test file(s) (older than {max_age_hours} hours)")
    
    return deleted_count


def verify_test_results() -> bool:
    """
    Verify test results in database.
    
    Returns:
        True if verification passes
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST PIPELINE: VERIFICATION")
    logger.info("=" * 80)
    
    try:
        from sqlalchemy import create_engine, text
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.warning("DATABASE_URL not set, skipping verification")
            return True
        
        engine = create_engine(database_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Count test records
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT source_file) as unique_files,
                    COUNT(DISTINCT h3_index) as unique_hexagons,
                    MIN(occurred_at) as earliest_crime,
                    MAX(occurred_at) as latest_crime
                FROM crimes
                WHERE source_file LIKE 'test_%'
            """))
            
            row = result.fetchone()
            if row:
                total, files, hexagons, earliest, latest = row
                logger.info(f"Total test records: {total:,}")
                logger.info(f"Unique source files: {files}")
                logger.info(f"Unique H3 hexagons: {hexagons:,}")
                if earliest:
                    logger.info(f"Earliest crime: {earliest}")
                if latest:
                    logger.info(f"Latest crime: {latest}")
                
                if total > 0:
                    logger.success("✅ Test data found in database")
                    return True
                else:
                    logger.warning("⚠️  No test data found in database")
                    return False
            else:
                logger.warning("⚠️  Could not query database")
                return False
                
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        logger.exception(e)
        return False


def main():
    """Run test pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline with small dataset")
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Maximum number of rows to process (default: 1000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="major-crime-indicators",
        help="Dataset to test (default: major-crime-indicators)",
    )
    parser.add_argument(
        "--skip-bronze",
        action="store_true",
        help="Skip Bronze layer (use existing test file)",
    )
    parser.add_argument(
        "--skip-silver",
        action="store_true",
        help="Skip Silver layer",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Skip Gold layer",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification (skip all processing)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old test files before running (files older than 24 hours)",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Cleanup old test files if requested
    if args.cleanup:
        cleanup_old_test_files(max_age_hours=24)
    
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("TEST PIPELINE - STARTING")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Max rows: {args.rows:,}")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    # Bronze layer
    if not args.verify_only:
        if not args.skip_bronze:
            try:
                test_file = download_test_data(args.dataset, args.rows)
                results["Bronze Download"] = True
                
                # Process through Bronze validation
                results["Bronze Process"] = process_bronze_test(test_file, args.dataset)
            except Exception as e:
                logger.error(f"❌ Bronze layer failed: {e}")
                logger.exception(e)
                results["Bronze"] = False
        else:
            logger.info("Skipping Bronze layer (--skip-bronze)")
            results["Bronze"] = True
        
        # Silver layer
        if not args.skip_silver:
            results["Silver"] = process_silver_test()
        else:
            logger.info("Skipping Silver layer (--skip-silver)")
            results["Silver"] = True
        
        # Gold layer
        if not args.skip_gold:
            results["Gold"] = process_gold_test()
        else:
            logger.info("Skipping Gold layer (--skip-gold)")
            results["Gold"] = True
    else:
        logger.info("Verification only mode (--verify-only)")
    
    # Verification
    results["Verification"] = verify_test_results()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    for phase, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{phase}: {status}")
    
    logger.info("")
    
    # Exit code
    all_success = all(results.values())
    if all_success:
        logger.success("🎉 Test pipeline completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check the database for test records")
        logger.info("  2. Review logs for any warnings")
        logger.info("  3. Run full pipeline: python scripts/run_pipeline.py")
        return 0
    else:
        logger.error("💥 Test pipeline had failures!")
        logger.info("")
        logger.info("Review the errors above and fix issues before running full pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
