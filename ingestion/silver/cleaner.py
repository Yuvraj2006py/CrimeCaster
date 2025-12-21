"""
Silver layer: Data cleaning and normalization.

Cleans raw CSV data from Bronze layer:
- Removes duplicates
- Handles missing values
- Validates coordinates
- Standardizes formats
- Transforms to target schema
"""

import os
import sys
from pathlib import Path
from datetime import datetime, time
from typing import Optional, Tuple
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Toronto coordinate bounds (approximate)
TORONTO_LAT_MIN = 43.5
TORONTO_LAT_MAX = 43.9
TORONTO_LON_MIN = -79.8
TORONTO_LON_MAX = -79.0


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def get_data_directories() -> Tuple[Path, Path]:
    """Get raw and silver data directories."""
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    silver_dir = project_root / "data" / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, silver_dir


def find_latest_bronze_file(raw_dir: Path) -> Optional[Path]:
    """Find the most recent CSV file in raw directory."""
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    # Sort by modification time, get most recent
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest


def find_all_bronze_files(raw_dir: Path) -> list[Path]:
    """Find all CSV files in raw directory."""
    csv_files = list(raw_dir.glob("*.csv"))
    # Sort by modification time (oldest first for consistent processing)
    return sorted(csv_files, key=lambda p: p.stat().st_mtime)


def load_bronze_csv(csv_path: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV from Bronze layer.
    
    Args:
        csv_path: Path to CSV file
        chunk_size: If provided, process in chunks (for large files)
        
    Returns:
        DataFrame with raw data
    """
    try:
        logger.info(f"Loading CSV from Bronze layer: {csv_path.name}")
        
        # Try different encodings
        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        df = None
        
        for encoding in encodings:
            try:
                if chunk_size:
                    # Process in chunks for very large files
                    chunks = []
                    for chunk in pd.read_csv(
                        csv_path,
                        encoding=encoding,
                        chunksize=chunk_size,
                        low_memory=False,
                    ):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(
                        csv_path,
                        encoding=encoding,
                        low_memory=False,
                    )
                logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Failed to load CSV with any encoding")
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def validate_and_clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean coordinate data.
    
    Args:
        df: DataFrame with LAT_WGS84 and LONG_WGS84 columns
        
    Returns:
        DataFrame with invalid coordinates removed
    """
    initial_count = len(df)
    
    # Check for required columns
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        if col.upper() in ["LAT_WGS84", "LAT", "LATITUDE"]:
            lat_col = col
        if col.upper() in ["LONG_WGS84", "LONG", "LONGITUDE", "LON"]:
            lon_col = col
    
    if not lat_col or not lon_col:
        raise ValueError(f"Coordinate columns not found. Found: {list(df.columns)}")
    
    # Convert to numeric, coerce errors to NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    
    # Filter valid Toronto coordinates
    valid_mask = (
        (df[lat_col] >= TORONTO_LAT_MIN)
        & (df[lat_col] <= TORONTO_LAT_MAX)
        & (df[lon_col] >= TORONTO_LON_MIN)
        & (df[lon_col] <= TORONTO_LON_MAX)
        & df[lat_col].notna()
        & df[lon_col].notna()
    )
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count:,} rows with invalid coordinates")
        logger.info(
            f"Coordinate bounds: lat [{TORONTO_LAT_MIN}, {TORONTO_LAT_MAX}], "
            f"lon [{TORONTO_LON_MIN}, {TORONTO_LON_MAX}]"
        )
    
    df_clean = df[valid_mask].copy()
    removed = initial_count - len(df_clean)
    
    if removed > 0:
        logger.info(f"Removed {removed:,} rows with invalid coordinates ({removed/initial_count*100:.2f}%)")
    
    return df_clean


def parse_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Parse occurrence datetime from OCC_DATE and OCC_HOUR.
    
    Args:
        df: DataFrame with OCC_DATE and OCC_HOUR columns
        
    Returns:
        Series with datetime objects
    """
    # Find date and hour columns
    date_col = None
    hour_col = None
    
    for col in df.columns:
        col_upper = col.upper()
        if "OCC_DATE" in col_upper or ("OCC" in col_upper and "DATE" in col_upper):
            date_col = col
        if "OCC_HOUR" in col_upper or ("OCC" in col_upper and "HOUR" in col_upper):
            hour_col = col
    
    if not date_col:
        raise ValueError(f"Date column not found. Available: {list(df.columns)}")
    
    # Parse date
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        raise
    
    # Add hour if available
    if hour_col and hour_col in df.columns:
        hours = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
        hours = hours.clip(0, 23)  # Ensure valid hour range
        # Combine date and hour
        datetimes = dates + pd.to_timedelta(hours, unit="h")
    else:
        datetimes = dates
    
    # Check for invalid dates
    invalid_dates = datetimes.isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates:,} rows with invalid dates")
    
    return datetimes


def standardize_crime_types(df: pd.DataFrame) -> pd.Series:
    """
    Standardize crime type names.
    
    Uses MCI_CATEGORY (high-level category) rather than detailed OFFENCE.
    
    Args:
        df: DataFrame with MCI_CATEGORY column
        
    Returns:
        Series with standardized crime types
    """
    # Find crime type column - prefer MCI_CATEGORY over OFFENCE
    crime_col = None
    for col in df.columns:
        if col.upper() == "MCI_CATEGORY":
            crime_col = col
            break
    
    # Fallback to other columns if MCI_CATEGORY not found
    if not crime_col:
        for col in df.columns:
            if col.upper() in ["CRIME_TYPE", "MAJOR_CRIME_INDICATOR"]:
                crime_col = col
                break
    
    if not crime_col:
        raise ValueError(f"Crime type column (MCI_CATEGORY) not found. Available: {list(df.columns)}")
    
    crime_types = df[crime_col].astype(str).str.strip()
    
    # Standardize common variations to match expected categories
    # Map to exact expected values (case-sensitive)
    replacements = {
        "break and enter": "Break and Enter",
        "break & enter": "Break and Enter",
        "b&e": "Break and Enter",
        "break and enter": "Break and Enter",  # Already correct
        "auto theft": "Auto Theft",
        "theft over": "Theft Over",
        "theftover": "Theft Over",
        "assault": "Assault",
        "robbery": "Robbery",
    }
    
    # Normalize to lowercase first for matching
    crime_types_lower = crime_types.str.lower().str.strip()
    
    # Apply replacements
    for old, new in replacements.items():
        crime_types = crime_types.where(crime_types_lower != old.lower(), new)
    
    # For any remaining, use title case but fix "and" to lowercase
    crime_types = crime_types.str.title()
    crime_types = crime_types.str.replace(" And ", " and ", regex=False)
    
    # Ensure we have valid categories
    valid_categories = ["Assault", "Robbery", "Break and Enter", "Auto Theft", "Theft Over"]
    invalid = ~crime_types.isin(valid_categories)
    if invalid.sum() > 0:
        logger.warning(f"Found {invalid.sum()} rows with unexpected crime types: {crime_types[invalid].unique()}")
    
    return crime_types


def clean_neighbourhood(df: pd.DataFrame) -> pd.Series:
    """
    Clean and standardize neighbourhood names.
    
    Args:
        df: DataFrame with neighbourhood column
        
    Returns:
        Series with cleaned neighbourhood names
    """
    # Try to find neighbourhood column
    hood_col = None
    for col in df.columns:
        if "NEIGHBOURHOOD" in col.upper() or "NEIGHBORHOOD" in col.upper():
            hood_col = col
            break
    
    if not hood_col:
        logger.warning("Neighbourhood column not found, returning empty series")
        return pd.Series([None] * len(df))
    
    neighbourhoods = df[hood_col].astype(str).str.strip()
    
    # Remove common prefixes/suffixes
    neighbourhoods = neighbourhoods.str.replace(r"^\d+\s*", "", regex=True)  # Remove leading numbers
    neighbourhoods = neighbourhoods.str.replace(r"\s*\(\d+\)$", "", regex=True)  # Remove trailing (number)
    
    # Replace empty strings with None
    neighbourhoods = neighbourhoods.replace("", None).replace("nan", None)
    
    return neighbourhoods


def clean_premise_type(df: pd.DataFrame) -> pd.Series:
    """
    Clean and standardize premise type.
    
    Args:
        df: DataFrame with PREMISES_TYPE column
        
    Returns:
        Series with cleaned premise types
    """
    # Find premise type column
    premise_col = None
    for col in df.columns:
        if "PREMISES_TYPE" in col.upper() or "PREMISE_TYPE" in col.upper():
            premise_col = col
            break
    
    if not premise_col:
        logger.warning("Premise type column not found, returning empty series")
        return pd.Series([None] * len(df))
    
    premise_types = df[premise_col].astype(str).str.strip()
    
    # Standardize common values
    premise_types = premise_types.replace("", None).replace("nan", None)
    
    return premise_types


def remove_duplicates(df: pd.DataFrame, unique_id_col: str = "EVENT_UNIQUE_ID") -> pd.DataFrame:
    """
    Remove duplicate records.
    
    Args:
        df: DataFrame to deduplicate
        unique_id_col: Column name for unique identifier
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    
    if unique_id_col not in df.columns:
        logger.warning(f"Unique ID column '{unique_id_col}' not found, using all columns for deduplication")
        df_clean = df.drop_duplicates()
    else:
        # Remove duplicates based on unique ID
        df_clean = df.drop_duplicates(subset=[unique_id_col], keep="first")
    
    removed = initial_count - len(df_clean)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate rows ({removed/initial_count*100:.2f}%)")
    
    return df_clean


def extract_dataset_type_from_filename(filename: str) -> Optional[str]:
    """
    Extract dataset type from filename.
    
    Filenames are in format: {dataset_type}_{timestamp}.csv
    e.g., 'major-crime-indicators_20251220_123456.csv'
    
    Args:
        filename: CSV filename
        
    Returns:
        Dataset type/key or None if not found
    """
    # Remove .csv extension
    name_without_ext = filename.replace(".csv", "")
    
    # Check if filename matches pattern: {dataset_type}_{timestamp}
    # Split by underscore and check if first part matches known dataset keys
    from ingestion.bronze.dataset_config import TORONTO_DATASETS
    
    parts = name_without_ext.split("_")
    if len(parts) >= 2:
        # Try to match first part(s) as dataset key
        # Some dataset keys have underscores, so try progressively longer matches
        for i in range(1, len(parts) + 1):
            potential_key = "_".join(parts[:i])
            if potential_key in TORONTO_DATASETS:
                return potential_key
    
    # Fallback: check if it's the old format 'toronto_crime_{timestamp}'
    if name_without_ext.startswith("toronto_crime_"):
        # Default to major-crime-indicators for legacy files
        return "major-crime-indicators"
    
    logger.warning(f"Could not extract dataset_type from filename: {filename}")
    return None
    return None


def transform_to_schema(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Transform cleaned data to match target schema.
    
    Args:
        df: Cleaned DataFrame from Bronze layer
        source_file: Original CSV filename
        
    Returns:
        DataFrame with schema columns
    """
    logger.info("Transforming data to target schema...")
    
    # Find coordinate columns
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ["LAT_WGS84", "LAT", "LATITUDE"]:
            lat_col = col
        if col_upper in ["LONG_WGS84", "LONG", "LONGITUDE", "LON"]:
            lon_col = col
    
    if not lat_col or not lon_col:
        raise ValueError("Coordinate columns not found")
    
    # Extract dataset_type from filename
    dataset_type = extract_dataset_type_from_filename(source_file)
    if dataset_type:
        logger.info(f"Extracted dataset_type: {dataset_type}")
    else:
        logger.warning(f"Could not extract dataset_type from filename: {source_file}")
    
    # Create transformed DataFrame
    transformed = pd.DataFrame()
    
    # Parse datetime
    transformed["occurred_at"] = parse_datetime(df)
    
    # Coordinates
    transformed["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    transformed["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    
    # Crime type
    transformed["crime_type"] = standardize_crime_types(df)
    
    # Neighbourhood
    transformed["neighbourhood"] = clean_neighbourhood(df)
    
    # Premise type
    transformed["premise_type"] = clean_premise_type(df)
    
    # Source file
    transformed["source_file"] = source_file
    
    # Dataset type (extracted from filename)
    transformed["dataset_type"] = dataset_type
    
    # Remove rows with missing critical fields
    critical_mask = (
        transformed["occurred_at"].notna()
        & transformed["latitude"].notna()
        & transformed["longitude"].notna()
        & transformed["crime_type"].notna()
    )
    
    invalid_count = (~critical_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count:,} rows with missing critical fields")
        transformed = transformed[critical_mask].copy()
    
    logger.info(f"Transformed to {len(transformed):,} valid rows")
    
    return transformed


def save_silver_data(df: pd.DataFrame, output_path: Path):
    """
    Save cleaned data to Silver layer.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save cleaned CSV
    """
    try:
        logger.info(f"Saving cleaned data to: {output_path}")
        df.to_csv(output_path, index=False)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved {len(df):,} rows ({file_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {e}")
        raise


def process_single_file(
    csv_path: Path,
    output_file: Optional[str] = None,
    skip_validation: bool = False,
) -> bool:
    """
    Process a single Bronze CSV file through Silver layer.
    
    Args:
        csv_path: Path to Bronze CSV file
        output_file: Path to save cleaned CSV (if None, auto-generates)
        skip_validation: Skip coordinate validation (not recommended)
        
    Returns:
        True if successful, False otherwise
    """
    source_file = csv_path.name
    raw_dir, silver_dir = get_data_directories()
    
    logger.info(f"Processing: {csv_path.name}")
    
    # Load Bronze data
    try:
        df = load_bronze_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load Bronze CSV {csv_path.name}: {e}")
        return False
    
    initial_count = len(df)
    logger.info(f"Starting with {initial_count:,} rows")
    
    # Step 1: Remove duplicates
    logger.info("Step 1: Removing duplicates...")
    df = remove_duplicates(df)
    
    # Step 2: Validate and clean coordinates
    if not skip_validation:
        logger.info("Step 2: Validating coordinates...")
        df = validate_and_clean_coordinates(df)
    else:
        logger.warning("Skipping coordinate validation (not recommended)")
    
    # Step 3: Transform to target schema
    logger.info("Step 3: Transforming to target schema...")
    try:
        df_clean = transform_to_schema(df, source_file)
    except Exception as e:
        logger.error(f"Transformation failed for {csv_path.name}: {e}")
        return False
    
    # Step 4: Save cleaned data
    if output_file:
        output_path = Path(output_file)
    else:
        # Generate output filename based on source file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract dataset name from source file (e.g., "major-crime-indicators_20251221_110200.csv" -> "major-crime-indicators")
        dataset_name = csv_path.stem.split("_")[0] if "_" in csv_path.stem else csv_path.stem
        output_path = silver_dir / f"cleaned_{dataset_name}_{timestamp}.csv"
    
    try:
        save_silver_data(df_clean, output_path)
    except Exception as e:
        logger.error(f"Failed to save cleaned data for {csv_path.name}: {e}")
        return False
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Silver Layer Complete for {csv_path.name}!")
    logger.info("=" * 60)
    logger.info(f"Initial rows: {initial_count:,}")
    logger.info(f"Final rows: {len(df_clean):,}")
    logger.info(f"Removed: {initial_count - len(df_clean):,} ({((initial_count - len(df_clean))/initial_count*100):.2f}%)")
    logger.info(f"Cleaned data saved to: {output_path}")
    
    return True


def main(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    skip_validation: bool = False,
    process_all: bool = False,
):
    """
    Main Silver layer cleaning function.
    
    Args:
        input_file: Path to Bronze CSV file (if None, uses latest or all files)
        output_file: Path to save cleaned CSV (if None, auto-generates)
        skip_validation: Skip coordinate validation (not recommended)
        process_all: If True, process all Bronze files (ignores input_file)
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Silver Layer: Data Cleaning")
    logger.info("=" * 60)
    
    # Get directories
    raw_dir, silver_dir = get_data_directories()
    
    # Process all files if requested
    if process_all:
        csv_files = find_all_bronze_files(raw_dir)
        if not csv_files:
            logger.error(f"No CSV files found in {raw_dir}")
            sys.exit(1)
        
        logger.info(f"Processing {len(csv_files)} Bronze files...")
        logger.info("")
        
        success_count = 0
        failed_count = 0
        
        for csv_path in csv_files:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Processing file {csv_files.index(csv_path) + 1}/{len(csv_files)}")
            logger.info("=" * 60)
            
            if process_single_file(csv_path, skip_validation=skip_validation):
                success_count += 1
            else:
                failed_count += 1
                logger.warning(f"Failed to process {csv_path.name}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Silver Layer: All Files Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"Successfully processed: {success_count}/{len(csv_files)} files")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count}/{len(csv_files)} files")
        logger.info(f"Next step: Run Gold layer to map H3 and load into database")
        return
    
    # Process single file
    # Determine input file
    if input_file:
        csv_path = Path(input_file)
        if not csv_path.exists():
            logger.error(f"Input file not found: {csv_path}")
            sys.exit(1)
    else:
        csv_path = find_latest_bronze_file(raw_dir)
        if not csv_path:
            logger.error(f"No CSV files found in {raw_dir}")
            sys.exit(1)
        logger.info(f"Using latest Bronze file: {csv_path.name}")
    
    if not process_single_file(csv_path, output_file, skip_validation):
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Silver layer data cleaning")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to Bronze CSV file (defaults to latest in data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save cleaned CSV (defaults to data/silver/cleaned_*.csv)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip coordinate validation (not recommended)",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all Bronze CSV files (ignores --input)",
    )
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        output_file=args.output,
        skip_validation=args.skip_validation,
        process_all=args.process_all,
    )

