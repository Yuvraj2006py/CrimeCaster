"""
Gold layer: H3 hexagon mapping and database loading.

Maps cleaned data from Silver layer to H3 hexagons and loads into PostgreSQL.
This is the final transformation step before data is available for ML training.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from io import StringIO
import csv
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
# Increased to 10000 for better performance with COPY method
# With COPY, we can use larger batches without hitting parameter limits
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


def save_to_parquet(df: pd.DataFrame, output_dir: Path, source_file: str) -> Path:
    """
    Save processed data to Parquet file for local ML training.
    
    Parquet format provides:
    - ~70% smaller file size than CSV
    - Fast columnar reads for ML training
    - Schema preservation
    - Compression support
    
    Args:
        df: Processed DataFrame with H3 indices and all columns
        output_dir: Directory to save Parquet files
        source_file: Original source file name (for naming)
        
    Returns:
        Path to saved Parquet file
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from source file (remove extension, add .parquet)
    base_name = Path(source_file).stem
    # Clean up filename (remove timestamps if present)
    parquet_path = output_dir / f"{base_name}.parquet"
    
    try:
        # Save as Parquet with compression
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',  # Fast compression, good balance
            index=False,
            write_statistics=True,  # For better query performance
        )
        
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Saved {len(df):,} rows to {parquet_path.name} ({file_size_mb:.1f} MB compressed)")
        
        return parquet_path
        
    except ImportError:
        logger.error("pyarrow not installed. Install with: pip install pyarrow")
        raise
    except Exception as e:
        logger.error(f"Error saving Parquet file: {e}")
        raise


def check_database_size(engine) -> Tuple[str, int]:
    """
    Check current database size and return formatted size and bytes.
    
    Args:
        engine: Database engine
        
    Returns:
        Tuple of (formatted_size_string, size_in_bytes)
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as db_size,
                    pg_database_size(current_database()) as db_size_bytes
            """))
            row = result.fetchone()
            if row:
                return row[0], row[1]  # formatted size, bytes
            return "Unknown", 0
    except Exception as e:
        logger.warning(f"Could not check database size: {e}")
        return "Unknown", 0


def can_proceed_with_inserts(engine, estimated_rows: int = 0) -> Tuple[bool, str]:
    """
    Check if database has enough space for inserts.
    
    Args:
        engine: Database engine
        estimated_rows: Estimated number of rows to insert (for size estimation)
        
    Returns:
        Tuple of (can_proceed, reason_message)
    """
    try:
        size_str, size_bytes = check_database_size(engine)
        
        # Neon free tier limit is 512 MB
        NEON_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB
        SAFE_THRESHOLD_BYTES = 450 * 1024 * 1024  # 450 MB (safe threshold to prevent hitting limit)
        
        if size_bytes == 0:
            # Can't determine size, allow but warn
            return True, "Could not determine database size, proceeding with caution"
        
        # Check if we're already at or over the limit
        if size_bytes >= NEON_LIMIT_BYTES:
            return False, f"Database is at or over the 512 MB limit ({size_str}). Cannot insert more data."
        
        # Check if we're too close to the limit (safe threshold)
        if size_bytes >= SAFE_THRESHOLD_BYTES:
            # Estimate if adding this data would exceed limit
            # Rough estimate: ~500 bytes per row (including indexes, overhead)
            estimated_bytes = estimated_rows * 500
            if size_bytes + estimated_bytes >= NEON_LIMIT_BYTES:
                return False, (
                    f"Database is at {size_str} (450+ MB). "
                    f"Adding {estimated_rows:,} rows (~{estimated_bytes / (1024*1024):.1f} MB) "
                    f"would exceed the 512 MB limit. Cannot proceed."
                )
            else:
                # Close but might fit
                return True, (
                    f"Database is at {size_str} (close to 512 MB limit). "
                    f"Proceeding with caution - may hit limit during insert."
                )
        
        # We're below the safe threshold, proceed
        return True, f"Database size: {size_str} (within safe limits)"
        
    except Exception as e:
        logger.warning(f"Error checking if inserts can proceed: {e}")
        # If we can't check, allow but warn
        return True, f"Could not verify database size: {e}. Proceeding with caution."


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
            pool_size=10,  # Maximum connections in pool (prevents exhaustion)
            max_overflow=20,  # Additional connections allowed beyond pool_size
            echo=False,  # Set to True for SQL debugging
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check database size and warn if approaching limit
        size_str, size_bytes = check_database_size(engine)
        if size_bytes > 0:
            logger.info(f"Database size: {size_str}")
            # Warn if approaching Neon free tier limit (512 MB)
            warning_threshold = 400 * 1024 * 1024  # 400 MB
            if size_bytes > warning_threshold:
                logger.warning(f"⚠️  Database size ({size_str}) is approaching the 512 MB Neon free tier limit!")
                logger.warning("Consider: 1) Upgrading plan, 2) Deleting old data, 3) Using a different provider")
        
        return engine
    except Exception as e:
        raise ValueError(f"Failed to connect to database: {e}")


def find_all_silver_files(silver_dir: Path, only_recent: bool = True, hours_threshold: int = 24) -> list[Path]:
    """
    Find all cleaned CSV files in Silver directory, excluding test files.
    
    Args:
        silver_dir: Directory containing Silver CSV files
        only_recent: If True, only return files from the last N hours (to avoid processing old files)
        hours_threshold: Number of hours to look back (default: 24, configurable for long-running pipelines)
        
    Returns:
        List of CSV file paths, sorted by modification time (newest first)
    """
    csv_files = list(silver_dir.glob("cleaned_*.csv"))
    
    # Exclude test files (files starting with "cleaned_test_")
    csv_files = [f for f in csv_files if not f.name.startswith("cleaned_test_")]
    
    if only_recent:
        # Only process files modified in the last N hours (configurable)
        cutoff_time = datetime.now() - timedelta(hours=hours_threshold)
        csv_files = [
            f for f in csv_files 
            if datetime.fromtimestamp(f.stat().st_mtime) > cutoff_time
        ]
        if csv_files:
            logger.info(f"Found {len(csv_files)} recent Silver files (last {hours_threshold} hours, excluding test files)")
        else:
            logger.warning(f"No recent Silver files found (last {hours_threshold} hours). Consider increasing hours_threshold if pipeline takes longer.")
    
    # Sort by modification time (newest first)
    return sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)


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


def load_silver_csv(csv_path: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load cleaned CSV from Silver layer with chunked reading for large files.
    
    Prevents OOM errors by processing large files in chunks (similar to Bronze layer).
    
    Args:
        csv_path: Path to Silver CSV file
        chunk_size: Number of rows per chunk (auto-determined if None)
        
    Returns:
        DataFrame with cleaned data
        
    Raises:
        ValueError: If CSV cannot be loaded or is invalid
    """
    try:
        logger.info(f"Loading Silver layer CSV: {csv_path.name}")
        
        # Check file size to determine if chunking is needed
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        use_chunking = file_size_mb > 50  # Same threshold as Bronze layer (50MB)
        
        # Validate required columns first (read header only)
        # Check if this is an aggregate dataset (no coordinates)
        header_df_sample = pd.read_csv(csv_path, nrows=1)
        is_aggregate = "has_coordinates" in header_df_sample.columns and header_df_sample["has_coordinates"].iloc[0] == False
        
        # Required columns depend on whether dataset has coordinates
        if is_aggregate:
            # Aggregate dataset - only require source_file and dataset_type
            required_columns = ["source_file", "dataset_type"]
        else:
            # Standard dataset with coordinates
            required_columns = [
                "occurred_at",
                "latitude",
                "longitude",
                "crime_type",
                "source_file",
            ]
        
        # Read header to validate columns
        header_df = pd.read_csv(csv_path, nrows=0)
        missing = [col for col in required_columns if col not in header_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Load CSV with or without chunking
        if use_chunking:
            chunk_size = chunk_size or 50000  # Default chunk size
            logger.info(f"Large file ({file_size_mb:.1f} MB), using chunked reading (chunk_size={chunk_size:,})...")
            chunks = []
            total_rows = 0
            
            try:
                for chunk in pd.read_csv(
                    csv_path,
                    chunksize=chunk_size,
                    parse_dates=["occurred_at"] if "occurred_at" in header_df.columns else [],
                    low_memory=False,
                ):
                    # Validate and clean each chunk
                    # Only filter on required columns (skip coordinate filtering for aggregate datasets)
                    initial_chunk_count = len(chunk)
                    if is_aggregate:
                        # For aggregate datasets, only filter if source_file is missing
                        chunk = chunk.dropna(subset=["source_file"])
                    else:
                        # For datasets with coordinates, filter nulls in required fields
                        chunk = chunk.dropna(subset=required_columns)
                    removed = initial_chunk_count - len(chunk)
                    if removed > 0:
                        logger.debug(f"Removed {removed:,} rows with null critical fields from chunk")
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Log progress for very large files
                    if len(chunks) % 10 == 0:
                        logger.info(f"Processed {len(chunks)} chunks ({total_rows:,} rows so far)...")
                
                # Combine all chunks
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded CSV with chunked reading: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as chunk_error:
                logger.error(f"Error during chunked reading: {chunk_error}")
                raise
        else:
            # Load entire file (small files)
            parse_dates_cols = ["occurred_at"] if "occurred_at" in header_df.columns else []
            df = pd.read_csv(
                csv_path,
                low_memory=False,
                parse_dates=parse_dates_cols,  # Parse datetime column if present
            )
            
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Check for nulls in critical fields (only for datasets with coordinates)
            if is_aggregate:
                # For aggregate datasets, only check source_file
                if "source_file" in df.columns:
                    initial_count = len(df)
                    df = df.dropna(subset=["source_file"])
                    removed = initial_count - len(df)
                    if removed > 0:
                        logger.warning(f"Removed {removed:,} rows with null source_file")
            else:
                # For datasets with coordinates, check all required fields
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
        
        # Validate data types (for both chunked and non-chunked)
        # Only validate occurred_at if it exists and dataset has coordinates
        if not is_aggregate and "occurred_at" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["occurred_at"]):
                logger.warning("occurred_at is not datetime, attempting conversion...")
                df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Silver CSV: {e}")
        raise


def map_to_h3_conditional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map to H3 hexagons only if coordinates are available.
    
    Args:
        df: DataFrame with latitude/longitude columns (or without)
        
    Returns:
        DataFrame with h3_index column (or None if no coordinates)
    """
    # Check if we have coordinates
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.info("No coordinate columns found - skipping H3 mapping")
        df["h3_index"] = None
        return df
    
    # Check if any rows have valid coordinates
    has_valid_coords = df["latitude"].notna().any() and df["longitude"].notna().any()
    
    if not has_valid_coords:
        logger.info("No valid coordinates found - skipping H3 mapping")
        df["h3_index"] = None
        return df
    
    # Map to H3
    return map_to_h3(df)


def map_to_h3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map latitude/longitude coordinates to H3 hexagon indices.
    
    Uses optimized vectorized operations for performance (10x faster than apply).
    
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
    
    # Optimized vectorized H3 mapping using pre-allocated array (faster than list append)
    logger.info("Computing H3 indices (vectorized)...")
    lats = df["latitude"].values.astype(float)
    lons = df["longitude"].values.astype(float)
    
    # Pre-allocate array for better performance (faster than list append)
    n = len(lats)
    h3_indices = [None] * n
    failed_count = 0
    
    # Optimized loop: direct indexing is faster than zip unpacking
    for i in range(n):
        try:
            h3_indices[i] = h3.latlng_to_cell(lats[i], lons[i], H3_RESOLUTION)
        except Exception:
            h3_indices[i] = None
            failed_count += 1
    
    df["h3_index"] = h3_indices
    
    # Check for failed H3 conversions
    if failed_count > 0:
        logger.warning(f"Failed to compute H3 index for {failed_count:,} rows")
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
    # Use view if columns are already in correct order, otherwise reorder
    if list(df.columns) == db_columns:
        df_db = df  # Already in correct order, use view (will copy later if needed)
    else:
        df_db = df[db_columns]  # Reorder columns, still a view
    
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
    Insert crimes data into database using PostgreSQL COPY (10-30x faster than executemany).
    
    Uses COPY FROM for bulk inserts with error handling and transaction management.
    
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
    
    logger.info(f"Inserting {total_rows:,} rows using COPY in batches of {batch_size:,}...")
    
    # Ensure columns are in correct order for COPY
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
    
    # Reorder DataFrame to match database column order (use view if columns already match)
    if list(df.columns) == db_columns:
        df_ordered = df  # No reordering needed, use view
    else:
        df_ordered = df[db_columns]  # Reorder columns, still a view
    
    # Split into batches
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df_ordered.iloc[start_idx:end_idx].copy()  # Copy only the batch slice
        
        try:
            # Use PostgreSQL COPY for bulk insert (much faster than executemany)
            # Use engine.connect() which properly handles connection pooling in parallel contexts
            with engine.connect() as conn:
                # Get the raw psycopg2 connection from SQLAlchemy connection
                raw_conn = conn.connection.dbapi_connection
                # Ensure we're in a transaction (not autocommit mode)
                # COPY requires explicit transaction management
                cursor = raw_conn.cursor()
                
                # Ultra-fast TSV preparation using vectorized pandas operations
                # This is 10-20x faster than iterrows()
                tsv_buffer = StringIO()
                
                # Prepare batch data: convert to proper formats for COPY
                batch_data = batch_df.copy()  # Work on copy to avoid modifying original
                
                # Handle datetime columns: convert to ISO format strings (vectorized)
                for col in db_columns:
                    if col == "occurred_at" and pd.api.types.is_datetime64_any_dtype(batch_data[col]):
                        # Vectorized datetime conversion: ensure timezone-aware, then format
                        # Convert to UTC if timezone-naive
                        if batch_data[col].dt.tz is None:
                            batch_data[col] = batch_data[col].dt.tz_localize('UTC')
                        else:
                            batch_data[col] = batch_data[col].dt.tz_convert('UTC')
                        # Convert to ISO format string (vectorized, much faster than apply)
                        # Use isoformat() via vectorized operation - PostgreSQL accepts ISO 8601 format
                        # Format: YYYY-MM-DDTHH:MM:SS+00:00 (ISO 8601)
                        # Note: strftime doesn't support timezone offset, so we format manually
                        batch_data[col] = (
                            batch_data[col].dt.strftime('%Y-%m-%dT%H:%M:%S') + '+00:00'
                        )
                        # Fill NaN with \N (after string operations to avoid type issues)
                        batch_data[col] = batch_data[col].fillna('\\N')
                    elif batch_data[col].dtype == 'object':
                        # Handle None/NaN for object columns (strings)
                        batch_data[col] = batch_data[col].fillna('\\N').astype(str)
                        # Escape special characters for COPY (vectorized string operations)
                        batch_data[col] = batch_data[col].str.replace('\\', '\\\\', regex=False)
                        batch_data[col] = batch_data[col].str.replace('\t', '\\t', regex=False)
                        batch_data[col] = batch_data[col].str.replace('\n', '\\n', regex=False)
                        batch_data[col] = batch_data[col].str.replace('\r', '\\r', regex=False)
                    else:
                        # Fill NaN with \N for numeric columns
                        batch_data[col] = batch_data[col].fillna('\\N').astype(str)
                
                # Write to TSV format using pandas to_csv (vectorized, very fast)
                batch_data.to_csv(
                    tsv_buffer,
                    sep='\t',
                    index=False,
                    header=False,
                    na_rep='\\N',
                )
                
                tsv_buffer.seek(0)
                
                # Use COPY FROM for bulk insert (tab-separated, NULL as \N)
                try:
                    cursor.copy_from(
                        tsv_buffer,
                        'crimes',
                        columns=db_columns,
                        null='\\N',
                        sep='\t'
                    )
                    # Commit the COPY transaction on the raw psycopg2 connection
                    # This is critical: COPY operations must be committed on the same connection
                    # that performed the COPY, not just through SQLAlchemy's connection wrapper
                    raw_conn.commit()
                    # SQLAlchemy connection will see the committed data on next query
                except Exception as copy_error:
                    conn.rollback()  # Rollback through SQLAlchemy connection
                    raise copy_error
                finally:
                    cursor.close()
            
            batch_success = len(batch_df)
            successful_rows += batch_success
            
            # Log progress (less frequently for performance - every 10 batches or last batch)
            if (batch_num + 1) % 10 == 0 or (batch_num + 1) == num_batches:
                progress = (batch_num + 1) / num_batches * 100
                logger.info(
                    f"Batch {batch_num + 1}/{num_batches} ({progress:.1f}%): "
                    f"Inserted {batch_success:,} rows (Total: {successful_rows:,}/{total_rows:,})"
                )
            
        except SQLAlchemyError as e:
            # Check for database size limit error (Neon free tier)
            error_str = str(e).lower()
            error_orig = str(e.orig) if hasattr(e, 'orig') else str(e)
            error_orig_lower = error_orig.lower()
            
            # Check for various forms of the disk full / size limit error
            is_disk_full = (
                "diskfull" in error_str or 
                "size limit" in error_str or 
                "512 mb" in error_str or
                "could not extend file" in error_orig_lower or
                "project size limit" in error_orig_lower or
                "neon.max_cluster_size" in error_orig_lower
            )
            
            if is_disk_full:
                logger.error(f"❌ DATABASE SIZE LIMIT EXCEEDED: {e}")
                logger.error("Your Neon database has reached the 512 MB free tier limit.")
                logger.error("")
                logger.error("Options to resolve:")
                logger.error("  1. Upgrade your Neon plan to get more storage")
                logger.error("  2. Delete old data:")
                logger.error("     DELETE FROM crimes WHERE occurred_at < NOW() - INTERVAL '1 year';")
                logger.error("  3. Use a different database provider with more storage")
                logger.error("")
                logger.error("Stopping batch insertion to prevent further errors.")
                remaining_rows = total_rows - successful_rows
                failed_rows += remaining_rows
                break
            
            logger.error(f"Error inserting batch {batch_num + 1}: {e}")
            logger.warning("Falling back to pandas to_sql for this batch...")
            
            # Fallback to pandas to_sql if COPY fails
            try:
                batch_df.to_sql(
                    "crimes",
                    engine,
                    if_exists="append",
                    index=False,
                    method=None,
                )
                successful_rows += len(batch_df)
                logger.info(f"Recovered {len(batch_df):,} rows using fallback method")
            except Exception as fallback_error:
                logger.error(f"Fallback insert also failed: {fallback_error}")
                failed_rows += len(batch_df)
        
        except Exception as e:
            # Check for database size limit error in generic exception handler too
            error_str = str(e).lower()
            error_orig = str(e.args[0]) if e.args else ""
            error_orig_lower = error_orig.lower()
            
            is_disk_full = (
                "diskfull" in error_str or 
                "size limit" in error_str or 
                "512 mb" in error_str or
                "could not extend file" in error_orig_lower or
                "project size limit" in error_orig_lower or
                "neon.max_cluster_size" in error_orig_lower
            )
            
            if is_disk_full:
                logger.error(f"❌ DATABASE SIZE LIMIT EXCEEDED: {e}")
                logger.error("Your Neon database has reached the 512 MB free tier limit.")
                logger.error("Stopping batch insertion to prevent further errors.")
                remaining_rows = total_rows - successful_rows
                failed_rows += remaining_rows
                break
            
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
    Verify that records were inserted correctly (optimized - single query).
    
    Args:
        engine: Database engine
        source_file: Source file name
        expected_count: Expected number of records
        
    Returns:
        True if verification passes
    """
    try:
        # Optimized: Use indexed query instead of DISTINCT (much faster for large datasets)
        # The source_file column is already indexed, so this query is efficient
        # Only check if the specific source_file exists (no need to list all)
        try:
            with engine.connect() as check_conn:
                # Quick existence check using index (faster than DISTINCT)
                check_result = check_conn.execute(
                    text("SELECT 1 FROM crimes WHERE source_file = :source_file LIMIT 1"),
                    {"source_file": source_file}
                )
                exists = check_result.fetchone() is not None
                if not exists:
                    logger.warning(f"Source file '{source_file}' not found in database. Records may not have been committed.")
        except Exception as e:
            logger.debug(f"Could not verify source_file existence: {e}")
        
        # Use engine.begin() to ensure we see committed data
        # Optimized: Combine 3 queries into 1 for better performance
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE h3_index IS NOT NULL) as with_h3,
                        COUNT(*) FILTER (WHERE geom IS NOT NULL) as with_geom
                    FROM crimes 
                    WHERE source_file = :source_file
                """),
                {"source_file": source_file}
            )
            row = result.fetchone()
            actual_count, h3_count, geom_count = row
            
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


def process_single_file(
    csv_path: Path,
    engine,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,
    verify: bool = True,
) -> bool:
    """
    Process a single Silver CSV file through Gold layer.
    
    Args:
        csv_path: Path to Silver CSV file
        engine: Database engine (will create thread-local engine if None)
        batch_size: Number of rows per batch insert
        skip_existing: Skip if records already exist for this file
        verify: Verify database insertion after completion
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {csv_path.name}")
    
    # Create thread-local engine for parallel processing (only if database loading is enabled)
    # If engine is None and database loading is enabled, create a new one
    load_to_database = os.getenv("LOAD_TO_DATABASE", "true").lower() == "true"
    if load_to_database:
        if engine is None:
            engine = get_database_connection()
    else:
        engine = None  # No database needed in local-only mode
    
    # Load Silver data first to get the actual source_file from the data
    try:
        df = load_silver_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load Silver CSV {csv_path.name}: {e}")
        return False
    
    # Extract source_file from the DataFrame (original Bronze file name)
    # All rows should have the same source_file, so take the first non-null value
    if "source_file" not in df.columns or df["source_file"].isna().all():
        logger.error(f"source_file column missing or all null in {csv_path.name}")
        return False
    source_file = df["source_file"].dropna().iloc[0] if len(df) > 0 else csv_path.name
    logger.info(f"Using source_file from data: {source_file}")
    
    # Check if dataset has coordinates
    has_coordinates = (
        "has_coordinates" in df.columns and df["has_coordinates"].iloc[0] == True
    ) or (
        "latitude" in df.columns and "longitude" in df.columns and
        df["latitude"].notna().any() and df["longitude"].notna().any()
    )
    
    if not has_coordinates:
        logger.info("Dataset does not have coordinates - processing as aggregate data")
        logger.info("Skipping H3 mapping and database insertion (saving to Parquet only)")
        
        # For aggregate datasets, we still want to save to Parquet for ML training
        # but skip database insertion (since crimes table requires coordinates)
        df["h3_index"] = None
        
        # Save to Parquet
        try:
            gold_dir = get_data_directories()[1]
            parquet_dir = gold_dir / "parquet"
            parquet_path = save_to_parquet(df, parquet_dir, source_file)
            logger.info(f"✅ Saved aggregate data to Parquet: {parquet_path.relative_to(gold_dir.parent.parent)}")
            logger.info(f"   Rows: {len(df):,}")
            logger.info("   Note: Aggregate datasets are saved to Parquet for ML feature engineering")
            return True
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            return False
    
    # Check if already loaded (only if database loading is enabled)
    if load_to_database and skip_existing and engine is not None:
        existing_count = check_existing_records(engine, source_file)
        if existing_count > 0:
            logger.warning(
                f"Found {existing_count:,} existing records for {source_file}. Skipping."
            )
            return True  # Consider it successful if already exists
    
    initial_count = len(df)
    logger.info(f"Starting with {initial_count:,} rows from Silver layer")
    
    # Map to H3 hexagons (only if we have coordinates)
    try:
        df = map_to_h3_conditional(df)
    except Exception as e:
        logger.error(f"Failed to map H3 indices for {csv_path.name}: {e}")
        return False
    
    # Prepare for database
    try:
        df_db = prepare_for_database(df)
    except Exception as e:
        logger.error(f"Failed to prepare data for database for {csv_path.name}: {e}")
        return False
    
    final_count = len(df_db)
    if final_count < initial_count:
        logger.warning(
            f"Data preparation reduced rows from {initial_count:,} to {final_count:,} "
            f"({(initial_count - final_count)/initial_count*100:.2f}% removed)"
        )
    
    # Save to Parquet for local ML training (always save, regardless of database setting)
    try:
        gold_dir = get_data_directories()[1]
        parquet_dir = gold_dir / "parquet"
        parquet_path = save_to_parquet(df_db, parquet_dir, source_file)
        logger.info(f"✅ Saved to Parquet: {parquet_path.relative_to(gold_dir.parent.parent)}")
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}")
        # Don't fail completely, but warn
        logger.warning("Continuing without Parquet save...")
    
    # Check if database loading is enabled
    load_to_database = os.getenv("LOAD_TO_DATABASE", "true").lower() == "true"
    
    if not load_to_database:
        logger.info("⏭️  Database loading disabled (LOAD_TO_DATABASE=false)")
        logger.info("   Data saved to Parquet files only for local ML training")
        successful_rows = len(df_db)
        failed_rows = 0
    else:
        # Check if we can proceed with inserts (database size check)
        can_proceed, reason = can_proceed_with_inserts(engine, estimated_rows=len(df_db))
        if not can_proceed:
            logger.error(f"❌ Cannot proceed with inserts: {reason}")
            logger.error("Skipping database insertion to prevent errors.")
            logger.info("   Data is still available in Parquet files for ML training")
            successful_rows = 0
            failed_rows = len(df_db)
        else:
            if "caution" in reason.lower() or "close" in reason.lower():
                logger.warning(f"⚠️  {reason}")
            
            # Get last timestamp for metadata
            last_timestamp = df_db["occurred_at"].max() if len(df_db) > 0 else None
            
            # Insert into database
            try:
                successful_rows, failed_rows = batch_insert_crimes(engine, df_db, batch_size=batch_size)
                
                logger.info("=" * 60)
                logger.info(f"Database Insertion Summary for {csv_path.name}")
                logger.info("=" * 60)
                logger.info(f"Successful: {successful_rows:,} rows")
                if failed_rows > 0:
                    logger.warning(f"Failed: {failed_rows:,} rows")
                if successful_rows + failed_rows > 0:
                    logger.info(f"Success rate: {successful_rows/(successful_rows+failed_rows)*100:.2f}%")
                
            except Exception as e:
                logger.error(f"Database insertion failed for {csv_path.name}: {e}")
                logger.info("   Data is still available in Parquet files for ML training")
                successful_rows = 0
                failed_rows = len(df_db)
            
            # Update ingestion metadata (only if we inserted to database)
            if successful_rows > 0:
                last_timestamp = df_db["occurred_at"].max() if len(df_db) > 0 else None
                update_ingestion_metadata(engine, source_file, successful_rows, last_timestamp)
            
            # Verify insertion (only if we inserted to database)
            if verify and successful_rows > 0:
                verify_database_insertion(engine, source_file, successful_rows)
    
    logger.info("=" * 60)
    logger.info(f"Gold Layer Complete for {csv_path.name}!")
    logger.info("=" * 60)
    
    return True


def main(
    input_file: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,
    verify: bool = True,
    process_all: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
):
    """
    Main Gold layer function: Map to H3 and load into database.
    
    Args:
        input_file: Path to Silver CSV file (if None, uses latest or all files)
        batch_size: Number of rows per batch insert
        skip_existing: Skip if records already exist for this file
        verify: Verify database insertion after completion
        process_all: If True, process all Silver files (ignores input_file)
        parallel: If True, process files in parallel (faster for multiple files)
        max_workers: Number of parallel workers (only used if parallel=True)
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Gold Layer: H3 Mapping and Storage")
    logger.info("=" * 60)
    
    # Check if database loading is enabled
    load_to_database = os.getenv("LOAD_TO_DATABASE", "true").lower() == "true"
    
    engine = None
    if load_to_database:
        logger.info("📊 Database mode: ENABLED")
        # Get database connection
        try:
            engine = get_database_connection()
            logger.info("Database connection established")
            
            # Check database size upfront and warn if approaching limit
            size_str, size_bytes = check_database_size(engine)
            if size_bytes > 0:
                NEON_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB
                SAFE_THRESHOLD_BYTES = 450 * 1024 * 1024  # 450 MB
                
                if size_bytes >= NEON_LIMIT_BYTES:
                    logger.error("❌ Database is at or over the 512 MB limit. Cannot proceed.")
                    logger.error("Please free up space before running the pipeline.")
                    logger.info("   Tip: Set LOAD_TO_DATABASE=false to use local-only mode")
                    sys.exit(1)
                elif size_bytes >= SAFE_THRESHOLD_BYTES:
                    logger.warning(f"⚠️  Database is at {size_str} (close to 512 MB limit).")
                    logger.warning("Some files may be skipped if they would exceed the limit.")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Set LOAD_TO_DATABASE=false to use local-only mode")
            sys.exit(1)
    else:
        logger.info("💾 Local-only mode: Database loading DISABLED")
        logger.info("   Data will be saved to Parquet files only")
        logger.info("   Perfect for ML training with unlimited data storage")
    
    # Get directories
    silver_dir, gold_dir = get_data_directories()
    
    # Process all files if requested
    if process_all:
        csv_files = find_all_silver_files(silver_dir, only_recent=True)
        if not csv_files:
            logger.error(f"No recent cleaned CSV files found in {silver_dir} (last 24 hours)")
            logger.info("Tip: If you want to process older files, modify find_all_silver_files() or use --input to specify files")
            sys.exit(1)
        
        logger.info(f"Processing {len(csv_files)} Silver files...")
        if parallel:
            logger.info(f"Using parallel processing with {max_workers} workers")
        logger.info("")
        
        success_count = 0
        failed_count = 0
        
        if parallel and len(csv_files) > 1:
            # Parallel processing for multiple files
            # Each thread gets its own engine to avoid connection pool conflicts
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files for processing (pass None for engine to create per-thread engines)
                future_to_file = {
                    executor.submit(process_single_file, csv_path, None, batch_size, skip_existing, verify): csv_path
                    for csv_path in csv_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_file):
                    csv_path = future_to_file[future]
                    try:
                        if future.result():
                            success_count += 1
                            logger.info(f"✅ Completed: {csv_path.name}")
                        else:
                            failed_count += 1
                            logger.warning(f"❌ Failed: {csv_path.name}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ Error processing {csv_path.name}: {e}")
        else:
            # Sequential processing
            for csv_path in csv_files:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"Processing file {csv_files.index(csv_path) + 1}/{len(csv_files)}")
                logger.info("=" * 60)
                
                if process_single_file(csv_path, engine, batch_size, skip_existing, verify):
                    success_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process {csv_path.name}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Gold Layer: All Files Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"Successfully processed: {success_count}/{len(csv_files)} files")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count}/{len(csv_files)} files")
        return
    
    # Process single file
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
    
    result = process_single_file(csv_path, engine, batch_size, skip_existing, verify)
    
    if not result:
        sys.exit(1)
    
    # Note: successful_rows is not available in this scope - it's only in process_single_file()
    # The detailed summary is already logged inside process_single_file(), so we just log success here
    logger.info("Data loaded successfully into database")
    logger.info(f"Data is now available for feature engineering and ML training")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gold layer: H3 mapping and database loading")
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all Silver CSV files (ignores --input)",
    )
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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple files in parallel (faster for multiple files)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, only used with --parallel)",
    )
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        verify=not args.no_verify,
        process_all=args.process_all,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )

