"""
Gold layer: H3 hexagon mapping and database loading.

Maps cleaned data from Silver layer to H3 hexagons and loads into PostgreSQL.
This is the final transformation step before data is available for ML training.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import h3
from loguru import logger
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# H3 resolution for crime data (resolution 9 = ~300m hexagons)
H3_RESOLUTION = 9

# Batch size for database inserts (optimize for performance)
BATCH_SIZE = 10000

# Toronto coordinate bounds for validation
TORONTO_LAT_MIN = 43.5
TORONTO_LAT_MAX = 43.9
TORONTO_LON_MIN = -79.8
TORONTO_LON_MAX = -79.0


def setup_logging():
    """Configure logging with detailed formatting."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def get_data_directories() -> Tuple[Path, Path]:
    """Get silver and gold data directories."""
    project_root = Path(__file__).parent.parent.parent
    silver_dir = project_root / "data" / "silver"
    gold_dir = project_root / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    return silver_dir, gold_dir


def get_database_connection():
    """
    Get database connection from environment variables.
    
    Returns:
        SQLAlchemy engine
        
    Raises:
        ValueError: If database connection cannot be established
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        # Try constructing from individual variables (fallback)
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD")
        
        if all([host, password]):
            database_url = f"postgresql://{db_user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(
                "DATABASE_URL not set. Please set DATABASE_URL in your .env file.\n"
                "Get it from your Neon dashboard: https://console.neon.tech"
            )
    
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for SQL debugging
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        raise ValueError(f"Failed to connect to database: {e}")


def find_latest_silver_file(silver_dir: Path) -> Optional[Path]:
    """
    Find the most recent CSV file in silver directory.
    
    Args:
        silver_dir: Path to silver data directory
        
    Returns:
        Path to latest CSV file or None if not found
    """
    csv_files = list(silver_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    # Sort by modification time, get most recent
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_silver_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load cleaned CSV from Silver layer.
    
    Args:
        csv_path: Path to Silver CSV file
        
    Returns:
        DataFrame with cleaned data
        
    Raises:
        ValueError: If CSV cannot be loaded or is invalid
    """
    try:
        logger.info(f"Loading Silver layer CSV: {csv_path.name}")
        
        # Load CSV
        df = pd.read_csv(
            csv_path,
            low_memory=False,
            parse_dates=["occurred_at"],  # Parse datetime column
        )
        
        # Validate required columns
        required_columns = [
            "occurred_at",
            "latitude",
            "longitude",
            "crime_type",
            "source_file",
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df["occurred_at"]):
            logger.warning("occurred_at is not datetime, attempting conversion...")
            df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
        
        # Check for nulls in critical fields
        critical_nulls = df[required_columns].isnull().sum()
        if critical_nulls.any():
            null_counts = critical_nulls[critical_nulls > 0]
            logger.warning(f"Found nulls in critical fields:\n{null_counts}")
            # Remove rows with nulls in critical fields
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            removed = initial_count - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed:,} rows with null critical fields")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Silver CSV: {e}")
        raise


def map_to_h3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map latitude/longitude coordinates to H3 hexagon indices.
    
    Uses vectorized operations for performance.
    
    Args:
        df: DataFrame with latitude and longitude columns
        
    Returns:
        DataFrame with added h3_index column
        
    Raises:
        ValueError: If coordinates are invalid
    """
    logger.info("Mapping coordinates to H3 hexagons (resolution {})...".format(H3_RESOLUTION))
    
    # Validate coordinates are within expected range
    invalid_coords = (
        (df["latitude"] < TORONTO_LAT_MIN)
        | (df["latitude"] > TORONTO_LAT_MAX)
        | (df["longitude"] < TORONTO_LON_MIN)
        | (df["longitude"] > TORONTO_LON_MAX)
        | df["latitude"].isna()
        | df["longitude"].isna()
    )
    
    if invalid_coords.any():
        invalid_count = invalid_coords.sum()
        logger.warning(
            f"Found {invalid_count:,} rows with invalid coordinates. These should have been filtered in Silver layer."
        )
        # Remove invalid coordinates
        df = df[~invalid_coords].copy()
        logger.warning(f"Removed {invalid_count:,} rows with invalid coordinates")
    
    # Vectorized H3 mapping
    def get_h3_index(row):
        """Get H3 index for a single row."""
        try:
            # Use latlng_to_cell (correct API for h3-py v4.x)
            return h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)
        except Exception as e:
            logger.error(f"Error computing H3 for lat={row['latitude']}, lon={row['longitude']}: {e}")
            return None
    
    # Apply H3 mapping
    logger.info("Computing H3 indices...")
    df["h3_index"] = df.apply(get_h3_index, axis=1)
    
    # Check for failed H3 conversions
    failed_h3 = df["h3_index"].isna().sum()
    if failed_h3 > 0:
        logger.warning(f"Failed to compute H3 index for {failed_h3:,} rows")
        df = df[df["h3_index"].notna()].copy()
    
    # Validate H3 indices
    unique_h3 = df["h3_index"].nunique()
    logger.info(f"Mapped to {unique_h3:,} unique H3 hexagons")
    
    return df


def prepare_for_database(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for database insertion.
    
    Ensures columns match database schema and data types are correct.
    
    Args:
        df: DataFrame with H3 indices
        
    Returns:
        DataFrame ready for database insertion
    """
    logger.info("Preparing data for database insertion...")
    
    # Select and order columns to match database schema
    db_columns = [
        "crime_type",
        "occurred_at",
        "latitude",
        "longitude",
        "neighbourhood",
        "premise_type",
        "h3_index",
        "source_file",
        "dataset_type",
    ]
    
    # Ensure all columns exist (fill missing with None)
    for col in db_columns:
        if col not in df.columns:
            df[col] = None
            logger.warning(f"Column {col} not found, filling with None")
    
    # Select only required columns in correct order
    df_db = df[db_columns].copy()
    
    # Ensure data types are correct
    # Convert occurred_at to timezone-aware datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_db["occurred_at"]):
        df_db["occurred_at"] = pd.to_datetime(df_db["occurred_at"], errors="coerce")
    
    # Ensure occurred_at is timezone-aware (UTC)
    if df_db["occurred_at"].dt.tz is None:
        df_db["occurred_at"] = df_db["occurred_at"].dt.tz_localize("UTC")
    else:
        df_db["occurred_at"] = df_db["occurred_at"].dt.tz_convert("UTC")
    
    # Convert numeric columns
    df_db["latitude"] = pd.to_numeric(df_db["latitude"], errors="coerce")
    df_db["longitude"] = pd.to_numeric(df_db["longitude"], errors="coerce")
    
    # Ensure string columns are strings (not objects)
    string_columns = ["crime_type", "neighbourhood", "premise_type", "h3_index", "source_file", "dataset_type"]
    for col in string_columns:
        df_db[col] = df_db[col].astype(str).replace("nan", None)
        # Truncate to max length if needed
        if col == "crime_type" and df_db[col].str.len().max() > 100:
            logger.warning(f"Truncating {col} to 100 characters")
            df_db[col] = df_db[col].str[:100]
        elif col in ["neighbourhood", "premise_type"] and df_db[col].str.len().max() > 100:
            logger.warning(f"Truncating {col} to 100 characters")
            df_db[col] = df_db[col].str[:100]
        elif col == "h3_index" and df_db[col].str.len().max() > 20:
            logger.warning(f"Truncating {col} to 20 characters")
            df_db[col] = df_db[col].str[:20]
        elif col == "source_file" and df_db[col].str.len().max() > 255:
            logger.warning(f"Truncating {col} to 255 characters")
            df_db[col] = df_db[col].str[:255]
        elif col == "dataset_type" and df_db[col].str.len().max() > 100:
            logger.warning(f"Truncating {col} to 100 characters")
            df_db[col] = df_db[col].str[:100]
    
    # Remove any remaining nulls in required fields
    required_fields = ["crime_type", "occurred_at", "latitude", "longitude", "h3_index", "source_file"]
    initial_count = len(df_db)
    df_db = df_db.dropna(subset=required_fields)
    removed = initial_count - len(df_db)
    if removed > 0:
        logger.warning(f"Removed {removed:,} rows with null required fields")
    
    logger.info(f"Prepared {len(df_db):,} rows for database insertion")
    
    return df_db


def check_existing_records(engine, source_file: str) -> int:
    """
    Check how many records already exist for this source file.
    
    Args:
        engine: Database engine
        source_file: Source file name
        
    Returns:
        Count of existing records
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM crimes WHERE source_file = :source_file"),
                {"source_file": source_file}
            )
            count = result.fetchone()[0]
            return count
    except Exception as e:
        logger.warning(f"Error checking existing records: {e}")
        return 0


def batch_insert_crimes(engine, df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """
    Insert crimes data into database in batches.
    
    Uses efficient bulk insert with error handling and transaction management.
    
    Args:
        engine: Database engine
        df: DataFrame with crimes data
        batch_size: Number of rows per batch
        
    Returns:
        Tuple of (successful_rows, failed_rows)
    """
    total_rows = len(df)
    successful_rows = 0
    failed_rows = 0
    
    logger.info(f"Inserting {total_rows:,} rows in batches of {batch_size:,}...")
    
    # Split into batches
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        try:
            # Use pandas to_sql for efficient bulk insert
            batch_df.to_sql(
                "crimes",
                engine,
                if_exists="append",
                index=False,
                method="multi",  # Use multi-row INSERT for speed
            )
            
            batch_success = len(batch_df)
            successful_rows += batch_success
            
            # Log progress
            progress = (batch_num + 1) / num_batches * 100
            logger.info(
                f"Batch {batch_num + 1}/{num_batches} ({progress:.1f}%): "
                f"Inserted {batch_success:,} rows (Total: {successful_rows:,}/{total_rows:,})"
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Error inserting batch {batch_num + 1}: {e}")
            failed_rows += len(batch_df)
            
            # Try individual inserts for this batch to identify problematic rows
            logger.warning(f"Attempting individual inserts for batch {batch_num + 1}...")
            individual_success = 0
            for idx, row in batch_df.iterrows():
                try:
                    row.to_frame().T.to_sql(
                        "crimes",
                        engine,
                        if_exists="append",
                        index=False,
                    )
                    individual_success += 1
                    successful_rows += 1
                except Exception as row_error:
                    logger.debug(f"Failed to insert row {idx}: {row_error}")
                    failed_rows += 1
            
            if individual_success > 0:
                logger.info(f"Recovered {individual_success:,} rows from failed batch")
        
        except Exception as e:
            logger.error(f"Unexpected error in batch {batch_num + 1}: {e}")
            failed_rows += len(batch_df)
    
    return successful_rows, failed_rows


def update_ingestion_metadata(
    engine, source_file: str, record_count: int, last_timestamp: Optional[datetime] = None
) -> bool:
    """
    Update ingestion metadata to mark Gold layer completion.
    
    Args:
        engine: Database engine
        source_file: Source file name
        record_count: Number of records inserted
        last_timestamp: Last crime timestamp (optional)
        
    Returns:
        True if successful
    """
    try:
        with engine.connect() as conn:
            # Update metadata to indicate Gold layer completion
            conn.execute(
                text("""
                    UPDATE ingestion_metadata
                    SET 
                        record_count = :record_count,
                        status = 'completed',
                        last_timestamp = :last_timestamp,
                        error_message = NULL
                    WHERE file_name = :source_file
                """),
                {
                    "source_file": source_file,
                    "record_count": record_count,
                    "last_timestamp": last_timestamp,
                }
            )
            conn.commit()
            logger.info(f"Updated ingestion metadata: {record_count:,} records loaded")
            return True
    except Exception as e:
        logger.error(f"Error updating ingestion metadata: {e}")
        return False


def verify_database_insertion(engine, source_file: str, expected_count: int) -> bool:
    """
    Verify that records were inserted correctly.
    
    Args:
        engine: Database engine
        source_file: Source file name
        expected_count: Expected number of records
        
    Returns:
        True if verification passes
    """
    try:
        with engine.connect() as conn:
            # Count records
            result = conn.execute(
                text("SELECT COUNT(*) FROM crimes WHERE source_file = :source_file"),
                {"source_file": source_file}
            )
            actual_count = result.fetchone()[0]
            
            # Check for H3 indices
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM crimes 
                    WHERE source_file = :source_file 
                    AND h3_index IS NOT NULL
                """),
                {"source_file": source_file}
            )
            h3_count = result.fetchone()[0]
            
            # Check for geometry
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM crimes 
                    WHERE source_file = :source_file 
                    AND geom IS NOT NULL
                """),
                {"source_file": source_file}
            )
            geom_count = result.fetchone()[0]
            
            logger.info("=" * 60)
            logger.info("Database Verification")
            logger.info("=" * 60)
            logger.info(f"Expected records: {expected_count:,}")
            logger.info(f"Actual records: {actual_count:,}")
            logger.info(f"Records with H3 index: {h3_count:,}")
            logger.info(f"Records with geometry: {geom_count:,}")
            
            if actual_count == expected_count and h3_count == expected_count and geom_count == expected_count:
                logger.info("[PASS] All records inserted correctly")
                return True
            else:
                logger.warning("[WARN] Record counts don't match expected values")
                return False
                
    except Exception as e:
        logger.error(f"Error verifying database insertion: {e}")
        return False


def main(
    input_file: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,
    verify: bool = True,
):
    """
    Main Gold layer function: Map to H3 and load into database.
    
    Args:
        input_file: Path to Silver CSV file (if None, uses latest)
        batch_size: Number of rows per batch insert
        skip_existing: Skip if records already exist for this file
        verify: Verify database insertion after completion
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Gold Layer: H3 Mapping and Database Loading")
    logger.info("=" * 60)
    
    # Get directories
    silver_dir, gold_dir = get_data_directories()
    
    # Determine input file
    if input_file:
        csv_path = Path(input_file)
        if not csv_path.exists():
            logger.error(f"Input file not found: {csv_path}")
            sys.exit(1)
    else:
        csv_path = find_latest_silver_file(silver_dir)
        if not csv_path:
            logger.error(f"No CSV files found in {silver_dir}")
            sys.exit(1)
        logger.info(f"Using latest Silver file: {csv_path.name}")
    
    source_file = csv_path.name
    
    # Get database connection
    try:
        engine = get_database_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Check if already loaded
    if skip_existing:
        existing_count = check_existing_records(engine, source_file)
        if existing_count > 0:
            logger.warning(
                f"Found {existing_count:,} existing records for {source_file}. "
                "Use --no-skip-existing to reload."
            )
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != "y":
                logger.info("Aborted by user")
                return
    
    # Load Silver data
    try:
        df = load_silver_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load Silver CSV: {e}")
        sys.exit(1)
    
    initial_count = len(df)
    logger.info(f"Starting with {initial_count:,} rows from Silver layer")
    
    # Map to H3 hexagons
    try:
        df = map_to_h3(df)
    except Exception as e:
        logger.error(f"Failed to map H3 indices: {e}")
        sys.exit(1)
    
    # Prepare for database
    try:
        df_db = prepare_for_database(df)
    except Exception as e:
        logger.error(f"Failed to prepare data for database: {e}")
        sys.exit(1)
    
    final_count = len(df_db)
    if final_count < initial_count:
        logger.warning(
            f"Data preparation reduced rows from {initial_count:,} to {final_count:,} "
            f"({(initial_count - final_count)/initial_count*100:.2f}% removed)"
        )
    
    # Get last timestamp for metadata
    last_timestamp = df_db["occurred_at"].max() if len(df_db) > 0 else None
    
    # Insert into database
    try:
        successful_rows, failed_rows = batch_insert_crimes(engine, df_db, batch_size=batch_size)
        
        logger.info("=" * 60)
        logger.info("Insertion Summary")
        logger.info("=" * 60)
        logger.info(f"Successful: {successful_rows:,} rows")
        if failed_rows > 0:
            logger.warning(f"Failed: {failed_rows:,} rows")
        logger.info(f"Success rate: {successful_rows/(successful_rows+failed_rows)*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        sys.exit(1)
    
    # Update ingestion metadata
    if successful_rows > 0:
        update_ingestion_metadata(engine, source_file, successful_rows, last_timestamp)
    
    # Verify insertion
    if verify and successful_rows > 0:
        verify_database_insertion(engine, source_file, successful_rows)
    
    logger.info("=" * 60)
    logger.info("Gold Layer Complete!")
    logger.info("=" * 60)
    logger.info(f"Loaded {successful_rows:,} crime records into database")
    logger.info(f"Data is now available for feature engineering and ML training")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gold layer: H3 mapping and database loading")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to Silver CSV file (defaults to latest in data/silver/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for database inserts (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip if records already exist",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip database verification after insertion",
    )
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        verify=not args.no_verify,
    )

