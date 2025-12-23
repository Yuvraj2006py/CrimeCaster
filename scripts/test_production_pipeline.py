"""
Test script to verify production pipeline works correctly.
Tests each layer individually to ensure everything is ready.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Configure logging."""
    # Fix Unicode encoding for Windows PowerShell
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

def test_bronze_files():
    """Test that Bronze layer finds production files (excluding test files)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Bronze Layer File Discovery")
    logger.info("=" * 80)
    
    try:
        from ingestion.silver.cleaner import find_all_bronze_files
        
        raw_dir = project_root / "data" / "raw"
        files = find_all_bronze_files(raw_dir, only_recent=True)
        
        # Check for test files
        test_files = [f for f in files if f.name.startswith("test_")]
        if test_files:
            logger.error(f"❌ Found {len(test_files)} test files in Bronze layer!")
            logger.error("Test files found:")
            for f in test_files:
                logger.error(f"  - {f.name}")
            return False
        
        logger.info(f"✅ Found {len(files)} production Bronze files (no test files)")
        if files:
            logger.info("Sample files:")
            for f in files[:3]:
                logger.info(f"  - {f.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bronze layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_silver_files():
    """Test that Silver layer finds production files (excluding test files)."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Silver Layer File Discovery")
    logger.info("=" * 80)
    
    try:
        from transformations.gold.h3_mapper import find_all_silver_files
        
        silver_dir = project_root / "data" / "silver"
        files = find_all_silver_files(silver_dir, only_recent=True)
        
        # Check for test files
        test_files = [f for f in files if f.name.startswith("cleaned_test_")]
        if test_files:
            logger.error(f"❌ Found {len(test_files)} test files in Silver layer!")
            logger.error("Test files found:")
            for f in test_files:
                logger.error(f"  - {f.name}")
            return False
        
        logger.info(f"✅ Found {len(files)} production Silver files (no test files)")
        if files:
            logger.info("Sample files:")
            for f in files[:3]:
                logger.info(f"  - {f.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Silver layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connection."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 3: Database Connection")
    logger.info("=" * 80)
    
    try:
        from sqlalchemy import create_engine, text
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            logger.error("❌ DATABASE_URL not set in .env file")
            return False
        
        engine = create_engine(database_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Check current record count
            result = conn.execute(text("SELECT COUNT(*) FROM crimes"))
            count = result.fetchone()[0]
            logger.info(f"✅ Database connected successfully")
            logger.info(f"   Current records in database: {count:,}")
            
            # Check for test records
            result = conn.execute(text("SELECT COUNT(*) FROM crimes WHERE source_file LIKE 'test_%'"))
            test_count = result.fetchone()[0]
            if test_count > 0:
                logger.warning(f"⚠️  Found {test_count:,} test records in database")
                logger.warning("   These will be excluded from new queries")
            else:
                logger.info("✅ No test records in database")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_silver_layer_processing():
    """Test that Silver layer can process production files."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 4: Silver Layer Processing (Dry Run)")
    logger.info("=" * 80)
    
    try:
        from ingestion.silver.cleaner import find_all_bronze_files
        
        raw_dir = project_root / "data" / "raw"
        files = find_all_bronze_files(raw_dir, only_recent=True)
        
        if not files:
            logger.warning("⚠️  No recent Bronze files found to process")
            logger.info("   This is OK - Bronze layer will download new files")
            return True
        
        logger.info(f"✅ Found {len(files)} Bronze files ready for processing")
        logger.info("   Silver layer will process these files (excluding test files)")
        
        # Check if files have coordinates (required for processing)
        from ingestion.silver.cleaner import load_bronze_csv
        from ingestion.silver.cleaner import build_column_mapping
        
        sample_file = files[0]
        logger.info(f"   Testing sample file: {sample_file.name}")
        
        try:
            df = load_bronze_csv(sample_file)
            col_map = build_column_mapping(df)
            
            if col_map.get('lat') and col_map.get('lon'):
                logger.info("   ✅ File has coordinate columns - can be processed")
            else:
                logger.warning("   ⚠️  File missing coordinate columns - will be skipped")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load sample file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Silver layer processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gold_layer_processing():
    """Test that Gold layer can process production files."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 5: Gold Layer Processing (Dry Run)")
    logger.info("=" * 80)
    
    try:
        from transformations.gold.h3_mapper import find_all_silver_files, load_silver_csv
        
        silver_dir = project_root / "data" / "silver"
        files = find_all_silver_files(silver_dir, only_recent=True)
        
        if not files:
            logger.warning("⚠️  No recent Silver files found to process")
            logger.info("   This is OK - Silver layer will create them first")
            return True
        
        logger.info(f"✅ Found {len(files)} Silver files ready for processing")
        
        # Test loading a sample file
        sample_file = files[0]
        logger.info(f"   Testing sample file: {sample_file.name}")
        
        try:
            df = load_silver_csv(sample_file)
            logger.info(f"   ✅ File loaded successfully: {len(df):,} rows")
            
            # Check for test source_file
            if 'source_file' in df.columns:
                source_files = df['source_file'].unique()
                test_sources = [s for s in source_files if str(s).startswith('test_')]
                if test_sources:
                    logger.error(f"   ❌ Found test source_file in data: {test_sources[0]}")
                    return False
                else:
                    logger.info(f"   ✅ All source_file values are production files")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load sample file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gold layer processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("PRODUCTION PIPELINE VERIFICATION TESTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Testing that pipeline will process only production data...")
    logger.info("")
    
    tests = [
        ("Bronze File Discovery", test_bronze_files),
        ("Silver File Discovery", test_silver_files),
        ("Database Connection", test_database_connection),
        ("Silver Layer Processing", test_silver_layer_processing),
        ("Gold Layer Processing", test_gold_layer_processing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.success("🎉 All tests passed! Pipeline is ready to run.")
        logger.info("")
        logger.info("Next step: Run the full production pipeline:")
        logger.info("  python scripts/run_pipeline.py")
        return 0
    else:
        logger.error("💥 Some tests failed! Fix issues before running pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

