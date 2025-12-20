"""
Unified data pipeline orchestrator.
Runs Bronze → Silver → Gold layers in sequence.

Usage:
    python scripts/run_pipeline.py

This script orchestrates the complete data pipeline:
1. Bronze Layer: Downloads all Toronto Open Data datasets
2. Silver Layer: Cleans and normalizes the data
3. Gold Layer: Maps to H3 hexagons and loads into database
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Configure logging for pipeline."""
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # Add file handler
    logger.add(
        logs_dir / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


def run_bronze_layer(all_datasets: bool = True, max_workers: int = 5) -> bool:
    """Run Bronze layer (download CSV files)."""
    logger.info("=" * 80)
    logger.info("PHASE 1: BRONZE LAYER - CSV Download")
    logger.info("=" * 80)
    
    try:
        from ingestion.bronze.csv_loader import main as bronze_main
        bronze_main(all_datasets=all_datasets, max_workers=max_workers)
        logger.success("✅ Bronze layer completed successfully")
        return True
    except SystemExit as e:
        # SystemExit(0) means success, SystemExit(1) means failure
        if e.code == 0:
            logger.success("✅ Bronze layer completed successfully")
            return True
        else:
            logger.error(f"❌ Bronze layer failed with exit code {e.code}")
            return False
    except Exception as e:
        logger.error(f"❌ Bronze layer failed: {e}")
        logger.exception(e)
        return False


def run_silver_layer() -> bool:
    """Run Silver layer (clean and normalize data)."""
    logger.info("=" * 80)
    logger.info("PHASE 2: SILVER LAYER - Data Cleaning")
    logger.info("=" * 80)
    
    try:
        from ingestion.silver.cleaner import main as silver_main
        silver_main()
        logger.success("✅ Silver layer completed successfully")
        return True
    except SystemExit as e:
        if e.code == 0:
            logger.success("✅ Silver layer completed successfully")
            return True
        else:
            logger.error(f"❌ Silver layer failed with exit code {e.code}")
            return False
    except Exception as e:
        logger.error(f"❌ Silver layer failed: {e}")
        logger.exception(e)
        return False


def run_gold_layer() -> bool:
    """Run Gold layer (H3 mapping and database loading)."""
    logger.info("=" * 80)
    logger.info("PHASE 3: GOLD LAYER - H3 Mapping & Database Loading")
    logger.info("=" * 80)
    
    try:
        from transformations.gold.h3_mapper import main as gold_main
        gold_main()
        logger.success("✅ Gold layer completed successfully")
        return True
    except SystemExit as e:
        if e.code == 0:
            logger.success("✅ Gold layer completed successfully")
            return True
        else:
            logger.error(f"❌ Gold layer failed with exit code {e.code}")
            return False
    except Exception as e:
        logger.error(f"❌ Gold layer failed: {e}")
        logger.exception(e)
        return False


def main():
    """Run complete data pipeline."""
    setup_logging()
    
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("QUARTERLY DATA PIPELINE - STARTING")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Run pipeline phases
    phases = [
        ("Bronze", lambda: run_bronze_layer()),
        ("Silver", run_silver_layer),
        ("Gold", run_gold_layer),
    ]
    
    results = {}
    for phase_name, phase_func in phases:
        logger.info("")
        success = phase_func()
        results[phase_name] = success
        
        if not success:
            logger.error(f"Pipeline stopped at {phase_name} layer")
            break
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration:.2f} minutes")
    logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    for phase_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{phase_name} Layer: {status}")
    
    logger.info("")
    
    # Exit code
    all_success = all(results.values())
    if all_success:
        logger.success("🎉 Pipeline completed successfully!")
        return 0
    else:
        logger.error("💥 Pipeline failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

