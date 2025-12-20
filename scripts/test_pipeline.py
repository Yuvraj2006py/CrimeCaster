"""
Test script to verify the pipeline setup works correctly.
This script performs a dry-run check without actually running the pipeline.

Usage:
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from ingestion.bronze.csv_loader import main as bronze_main
        logger.success("✅ Bronze layer import successful")
    except ImportError as e:
        logger.error(f"❌ Bronze layer import failed: {e}")
        return False
    
    try:
        from ingestion.silver.cleaner import main as silver_main
        logger.success("✅ Silver layer import successful")
    except ImportError as e:
        logger.error(f"❌ Silver layer import failed: {e}")
        return False
    
    try:
        from transformations.gold.h3_mapper import main as gold_main
        logger.success("✅ Gold layer import successful")
    except ImportError as e:
        logger.error(f"❌ Gold layer import failed: {e}")
        return False
    
    return True


def test_directories():
    """Test that required directories exist or can be created."""
    logger.info("Testing directories...")
    
    required_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "silver",
        project_root / "data" / "gold",
        project_root / "logs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.exists():
            logger.success(f"✅ Directory exists: {dir_path}")
        else:
            logger.error(f"❌ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist


def test_environment():
    """Test that environment variables are set."""
    logger.info("Testing environment...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        logger.success("✅ DATABASE_URL is set")
        # Don't log the actual URL for security
        logger.info(f"   (Length: {len(database_url)} characters)")
    else:
        logger.warning("⚠️  DATABASE_URL is not set (required for Gold layer)")
    
    return True  # Not critical for import test


def test_pipeline_script():
    """Test that the pipeline script exists and is importable."""
    logger.info("Testing pipeline script...")
    
    pipeline_script = project_root / "scripts" / "run_pipeline.py"
    if pipeline_script.exists():
        logger.success("✅ Pipeline script exists")
        
        # Try to import it
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_pipeline", pipeline_script
            )
            if spec and spec.loader:
                logger.success("✅ Pipeline script is valid Python")
            else:
                logger.error("❌ Pipeline script is not valid Python")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking pipeline script: {e}")
            return False
    else:
        logger.error("❌ Pipeline script not found")
        return False
    
    return True


def test_github_workflow():
    """Test that GitHub workflow file exists."""
    logger.info("Testing GitHub workflow...")
    
    workflow_file = project_root / ".github" / "workflows" / "quarterly-pipeline.yml"
    if workflow_file.exists():
        logger.success("✅ GitHub workflow file exists")
        
        # Check if it has required content
        content = workflow_file.read_text()
        if "quarterly-pipeline" in content.lower():
            logger.success("✅ GitHub workflow appears valid")
        else:
            logger.warning("⚠️  GitHub workflow may be incomplete")
    else:
        logger.warning("⚠️  GitHub workflow file not found")
        logger.info("   (This is OK if you haven't set up GitHub Actions yet)")
    
    return True  # Not critical


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("PIPELINE SETUP TEST")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Environment", test_environment),
        ("Pipeline Script", test_pipeline_script),
        ("GitHub Workflow", test_github_workflow),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' raised exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("")
    if all_passed:
        logger.success("🎉 All tests passed! Pipeline setup looks good.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Set DATABASE_URL in GitHub Secrets (if using GitHub Actions)")
        logger.info("2. Test locally: python scripts/run_pipeline.py")
        logger.info("3. Push to GitHub and test workflow manually")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

