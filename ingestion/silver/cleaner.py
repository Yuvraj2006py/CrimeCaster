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
from datetime import datetime, time, timedelta
from typing import Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


def find_all_bronze_files(raw_dir: Path, only_recent: bool = True) -> list[Path]:
    """
    Find all CSV files in raw directory, excluding test files.
    
    Args:
        raw_dir: Directory containing Bronze CSV files
        only_recent: If True, only return files from the last 24 hours (to avoid processing old files)
        
    Returns:
        List of CSV file paths, sorted by modification time (newest first)
    """
    csv_files = list(raw_dir.glob("*.csv"))
    
    # Exclude test files (files starting with "test_")
    csv_files = [f for f in csv_files if not f.name.startswith("test_")]
    
    if only_recent:
        # Only process files modified in the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        csv_files = [
            f for f in csv_files 
            if datetime.fromtimestamp(f.stat().st_mtime) > cutoff_time
        ]
        if csv_files:
            logger.info(f"Found {len(csv_files)} recent Bronze files (last 24 hours, excluding test files)")
    
    # Sort by modification time (newest first)
    return sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_bronze_csv(csv_path: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV or Excel file from Bronze layer.
    
    Args:
        csv_path: Path to CSV/Excel file
        chunk_size: If provided, process in chunks (for large files)
        
    Returns:
        DataFrame with raw data
    """
    try:
        logger.info(f"Loading CSV from Bronze layer: {csv_path.name}")
        
        # Check if file is Excel (.xlsx or .xls)
        is_excel = csv_path.suffix.lower() in ['.xlsx', '.xls']
        
        # Also check magic bytes for Excel files (PK\x03\x04)
        if not is_excel:
            try:
                with open(csv_path, 'rb') as f:
                    magic_bytes = f.read(4)
                    if magic_bytes == b'PK\x03\x04':
                        is_excel = True
            except Exception:
                pass
        
        if is_excel:
            # Load Excel file
            try:
                df = pd.read_excel(csv_path, engine='openpyxl')
                logger.info(f"Successfully loaded Excel file with openpyxl")
                logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.error(f"Error loading Excel file: {e}")
                raise
        
        # Try different encodings for CSV files
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
            except pd.errors.ParserError as e:
                # If CSV parsing fails, try with more lenient settings
                logger.warning(f"CSV parsing error with {encoding}, trying fallback method: {e}")
                try:
                    df = pd.read_csv(
                        csv_path,
                        encoding=encoding,
                        on_bad_lines='skip',
                        engine='python',
                    )
                    logger.info(f"Successfully loaded CSV with fallback method (encoding: {encoding})")
                    break
                except Exception:
                    continue
        
        if df is None:
            raise ValueError("Failed to load CSV with any encoding")
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def build_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Build column mapping once instead of searching multiple times.
    
    Args:
        df: DataFrame to map columns from
        
    Returns:
        Dict mapping target -> column_name (or None if not found)
    """
    col_upper_map = {col.upper(): col for col in df.columns}
    col_upper_list = list(col_upper_map.keys())
    
    mapping = {}
    
    # Find all columns we need in one pass
    for target, patterns in [
        ('date', ['OCC_DATE', 'OCC', 'DATE']),
        ('hour', ['OCC_HOUR', 'OCC', 'HOUR']),
        ('lat', ['LAT_WGS84', 'LAT', 'LATITUDE']),
        ('lon', ['LONG_WGS84', 'LONG', 'LONGITUDE', 'LON']),
        ('crime', ['MCI_CATEGORY', 'CRIME_TYPE', 'MAJOR_CRIME_INDICATOR', 'EVENT_TYPE']),
        ('neighbourhood', ['NEIGHBOURHOOD', 'NEIGHBORHOOD']),
        ('premise', ['PREMISES_TYPE', 'PREMISE_TYPE']),
        ('unique_id', ['EVENT_UNIQUE_ID', 'UNIQUE_ID', 'ID']),
    ]:
        found = False
        for pattern in patterns:
            # Try exact match first
            if pattern in col_upper_map:
                mapping[target] = col_upper_map[pattern]
                found = True
                break
            # Try substring match (for columns like NEIGHBOURHOOD_158)
            for col_upper in col_upper_list:
                if pattern in col_upper:
                    mapping[target] = col_upper_map[col_upper]
                    found = True
                    break
            if found:
                break
        
        if not found:
            mapping[target] = None
    
    return mapping


def validate_coordinates_vectorized(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    """
    Vectorized coordinate validation (single pass, optimized).
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Latitude column name
        lon_col: Longitude column name
        
    Returns:
        Boolean Series indicating valid coordinates
    """
    # Convert to numeric once
    lats = pd.to_numeric(df[lat_col], errors='coerce')
    lons = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Single vectorized mask (faster than multiple operations)
    valid_mask = (
        lats.between(TORONTO_LAT_MIN, TORONTO_LAT_MAX) &
        lons.between(TORONTO_LON_MIN, TORONTO_LON_MAX) &
        lats.notna() &
        lons.notna()
    )
    
    return valid_mask


def validate_and_clean_coordinates(df: pd.DataFrame, col_map: Optional[Dict[str, Optional[str]]] = None) -> Optional[pd.DataFrame]:
    """
    Validate and clean coordinate data (optimized with column mapping).
    
    Args:
        df: DataFrame with coordinate columns
        col_map: Optional pre-built column mapping (for performance)
        
    Returns:
        DataFrame with invalid coordinates removed, or None if columns not found
    """
    initial_count = len(df)
    
    # Use provided mapping or build one
    if col_map is None:
        col_map = build_column_mapping(df)
    
    lat_col = col_map.get('lat')
    lon_col = col_map.get('lon')
    
    if not lat_col or not lon_col:
        return None
    
    # Vectorized validation
    valid_mask = validate_coordinates_vectorized(df, lat_col, lon_col)
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count:,} rows with invalid coordinates")
    
    df_clean = df[valid_mask].copy()
    removed = initial_count - len(df_clean)
    
    if removed > 0:
        logger.info(f"Removed {removed:,} rows with invalid coordinates ({removed/initial_count*100:.2f}%)")
    
    return df_clean


def parse_datetime_fast(df: pd.DataFrame, col_map: Dict[str, Optional[str]]) -> pd.Series:
    """
    Parse occurrence datetime from date and hour columns (optimized with column mapping).
    
    Args:
        df: DataFrame with date and hour columns
        col_map: Pre-built column mapping
        
    Returns:
        Series with datetime objects
    """
    date_col = col_map.get('date')
    hour_col = col_map.get('hour')
    
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


def standardize_crime_types_fast(df: pd.DataFrame, crime_col: str) -> pd.Series:
    """
    Optimized crime type standardization using vectorized operations.
    
    Args:
        df: DataFrame with crime type column
        crime_col: Name of crime type column
        
    Returns:
        Series with standardized crime types
    """
    if not crime_col:
        raise ValueError(f"Crime type column not found. Available: {list(df.columns)}")
    
    crime_types = df[crime_col].astype(str).str.strip()
    
    # Standardize common variations to match expected categories
    replacements = {
        "break and enter": "Break and Enter",
        "break & enter": "Break and Enter",
        "b&e": "Break and Enter",
        "auto theft": "Auto Theft",
        "theft over": "Theft Over",
        "theftover": "Theft Over",
        "assault": "Assault",
        "robbery": "Robbery",
    }
    
    # Use map for vectorized replacement (much faster than multiple .str operations)
    crime_types_lower = crime_types.str.lower().str.strip()
    crime_types = crime_types_lower.map(replacements).fillna(crime_types)
    
    # Title case only unmatched values (optimization)
    mask = ~crime_types.isin(replacements.values())
    if mask.any():
        crime_types.loc[mask] = crime_types.loc[mask].str.title().str.replace(" And ", " and ", regex=False)
    
    # Ensure we have valid categories
    valid_categories = ["Assault", "Robbery", "Break and Enter", "Auto Theft", "Theft Over"]
    invalid = ~crime_types.isin(valid_categories)
    if invalid.sum() > 0:
        logger.warning(f"Found {invalid.sum()} rows with unexpected crime types: {crime_types[invalid].unique()}")
    
    return crime_types


def clean_neighbourhood_fast(df: pd.DataFrame, hood_col: Optional[str]) -> pd.Series:
    """
    Optimized neighbourhood cleaning (with pre-mapped column).
    
    Args:
        df: DataFrame with neighbourhood column
        hood_col: Name of neighbourhood column (or None)
        
    Returns:
        Series with cleaned neighbourhood names
    """
    if not hood_col:
        return pd.Series([None] * len(df), index=df.index)
    
    neighbourhoods = df[hood_col].astype(str).str.strip()
    
    # Chain operations for efficiency
    neighbourhoods = (
        neighbourhoods
        .str.replace(r"^\d+\s*", "", regex=True)  # Remove leading numbers
        .str.replace(r"\s*\(\d+\)$", "", regex=True)  # Remove trailing (number)
        .replace("", None)
        .replace("nan", None)
    )
    
    return neighbourhoods


def clean_premise_type_fast(df: pd.DataFrame, premise_col: Optional[str]) -> pd.Series:
    """
    Optimized premise type cleaning (with pre-mapped column).
    
    Args:
        df: DataFrame with premise type column
        premise_col: Name of premise type column (or None)
        
    Returns:
        Series with cleaned premise types
    """
    if not premise_col:
        return pd.Series([None] * len(df), index=df.index)
    
    premise_types = df[premise_col].astype(str).str.strip()
    premise_types = premise_types.replace("", None).replace("nan", None)
    
    return premise_types


def remove_duplicates_fast(df: pd.DataFrame, unique_id_col: Optional[str] = None) -> pd.DataFrame:
    """
    Fast duplicate removal using hash-based approach for large files.
    
    Args:
        df: DataFrame to deduplicate
        unique_id_col: Column name for unique identifier (or None to auto-detect)
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    
    # Auto-detect unique ID column if not provided
    if unique_id_col is None:
        for col in df.columns:
            if col.upper() in ["EVENT_UNIQUE_ID", "UNIQUE_ID", "ID"]:
                unique_id_col = col
                break
    
    if unique_id_col and unique_id_col in df.columns:
        # For very large DataFrames, use hash-based approach (O(n) vs O(n log n))
        if len(df) > 100000:
            # Use hash set for O(n) duplicate removal
            seen = set()
            mask = []
            for val in df[unique_id_col]:
                if val not in seen:
                    seen.add(val)
                    mask.append(True)
                else:
                    mask.append(False)
            df_clean = df[mask].copy()
        else:
            # For smaller DataFrames, use pandas drop_duplicates (optimized)
            df_clean = df.drop_duplicates(subset=[unique_id_col], keep="first")
    else:
        logger.warning(f"Unique ID column not found, using all columns for deduplication")
        df_clean = df.drop_duplicates(keep="first")
    
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


def transform_to_schema_flexible(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Transform data to flexible schema that works with or without coordinates.
    
    For datasets with coordinates: Uses standard schema with lat/lon
    For datasets without coordinates: Preserves all original columns
    
    Args:
        df: Cleaned DataFrame from Bronze layer
        source_file: Original CSV filename
        
    Returns:
        DataFrame with flexible schema
    """
    logger.info("Transforming data to flexible schema...")
    
    # Build column mapping once
    col_map = build_column_mapping(df)
    
    # Extract dataset_type from filename
    dataset_type = extract_dataset_type_from_filename(source_file)
    if dataset_type:
        logger.info(f"Extracted dataset_type: {dataset_type}")
    else:
        logger.warning(f"Could not extract dataset_type from filename: {source_file}")
    
    # Check if we have coordinates
    has_coordinates = col_map.get('lat') and col_map.get('lon')
    has_date = col_map.get('date')
    has_crime = col_map.get('crime')
    
    # Create transformed DataFrame
    transformed = pd.DataFrame(index=df.index)
    
    # Always add metadata columns
    transformed["source_file"] = source_file
    transformed["dataset_type"] = dataset_type
    # Set has_coordinates as a boolean Series (not scalar) to preserve type in CSV
    transformed["has_coordinates"] = pd.Series([has_coordinates] * len(df), index=df.index, dtype=bool)
    
    # Handle date column
    if has_date:
        transformed["occurred_at"] = parse_datetime_fast(df, col_map)
    else:
        # Try to find any date-like column
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
        if date_cols:
            try:
                transformed["occurred_at"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
                logger.info(f"Using date column: {date_cols[0]}")
            except:
                transformed["occurred_at"] = None
        else:
            transformed["occurred_at"] = None
            logger.warning("No date column found, setting occurred_at to None")
    
    if has_coordinates:
        # Standard schema with coordinates
        transformed["latitude"] = pd.to_numeric(df[col_map['lat']], errors="coerce")
        transformed["longitude"] = pd.to_numeric(df[col_map['lon']], errors="coerce")
        
        if has_crime:
            transformed["crime_type"] = standardize_crime_types_fast(df, col_map['crime'])
        else:
            transformed["crime_type"] = None
        
        transformed["neighbourhood"] = clean_neighbourhood_fast(df, col_map.get('neighbourhood'))
        transformed["premise_type"] = clean_premise_type_fast(df, col_map.get('premise'))
        
        # Filter invalid coordinates
        valid_mask = (
            transformed["occurred_at"].notna() &
            transformed["latitude"].notna() &
            transformed["longitude"].notna() &
            transformed["latitude"].between(TORONTO_LAT_MIN, TORONTO_LAT_MAX) &
            transformed["longitude"].between(TORONTO_LON_MIN, TORONTO_LON_MAX)
        )
        
        if has_crime:
            valid_mask = valid_mask & transformed["crime_type"].notna()
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count:,} rows with invalid/missing critical fields")
            transformed = transformed[valid_mask].copy()
    else:
        # No coordinates - preserve all original data
        logger.info("No coordinates found - preserving all original columns for aggregate data")
        transformed["latitude"] = None
        transformed["longitude"] = None
        transformed["crime_type"] = None
        transformed["neighbourhood"] = None
        transformed["premise_type"] = None
        
        # Preserve all original columns (for aggregate datasets) - use concat for performance
        raw_cols = {}
        for col in df.columns:
            if col not in transformed.columns:
                # Store original columns with 'raw_' prefix to avoid conflicts
                raw_cols[f"raw_{col}"] = df[col]
        
        # Add all raw columns at once using concat (avoids DataFrame fragmentation)
        if raw_cols:
            raw_df = pd.DataFrame(raw_cols, index=df.index)
            transformed = pd.concat([transformed, raw_df], axis=1)
    
    logger.info(f"Transformed to {len(transformed):,} rows")
    
    return transformed


def transform_to_schema_single_pass(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Transform cleaned data to match target schema in a single pass (optimized).
    
    All transformations are combined to minimize data passes and memory copies.
    
    Args:
        df: Cleaned DataFrame from Bronze layer
        source_file: Original CSV filename
        
    Returns:
        DataFrame with schema columns
    """
    logger.info("Transforming data to target schema (single-pass optimization)...")
    
    # Build column mapping once
    col_map = build_column_mapping(df)
    
    # Validate we have required columns
    if not col_map.get('lat') or not col_map.get('lon'):
        raise ValueError("Coordinate columns not found")
    
    if not col_map.get('date'):
        raise ValueError("Date column not found")
    
    if not col_map.get('crime'):
        raise ValueError("Crime type column not found")
    
    # Extract dataset_type from filename
    dataset_type = extract_dataset_type_from_filename(source_file)
    if dataset_type:
        logger.info(f"Extracted dataset_type: {dataset_type}")
    else:
        logger.warning(f"Could not extract dataset_type from filename: {source_file}")
    
    # Create transformed DataFrame with all operations in parallel (single pass)
    transformed = pd.DataFrame(index=df.index)
    
    # All transformations in parallel (vectorized)
    transformed["occurred_at"] = parse_datetime_fast(df, col_map)
    transformed["latitude"] = pd.to_numeric(df[col_map['lat']], errors="coerce")
    transformed["longitude"] = pd.to_numeric(df[col_map['lon']], errors="coerce")
    transformed["crime_type"] = standardize_crime_types_fast(df, col_map['crime'])
    transformed["neighbourhood"] = clean_neighbourhood_fast(df, col_map.get('neighbourhood'))
    transformed["premise_type"] = clean_premise_type_fast(df, col_map.get('premise'))
    transformed["source_file"] = source_file
    transformed["dataset_type"] = dataset_type
    
    # Filter invalid rows in one pass (combine coordinate validation and critical fields)
    valid_mask = (
        transformed["occurred_at"].notna() &
        transformed["latitude"].notna() &
        transformed["longitude"].notna() &
        transformed["crime_type"].notna() &
        transformed["latitude"].between(TORONTO_LAT_MIN, TORONTO_LAT_MAX) &
        transformed["longitude"].between(TORONTO_LON_MIN, TORONTO_LON_MAX)
    )
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count:,} rows with invalid/missing critical fields")
        transformed = transformed[valid_mask].copy()
    
    logger.info(f"Transformed to {len(transformed):,} valid rows")
    
    return transformed


def transform_to_schema(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Transform cleaned data to match target schema (wrapper for backward compatibility).
    
    Args:
        df: Cleaned DataFrame from Bronze layer
        source_file: Original CSV filename
        
    Returns:
        DataFrame with schema columns
    """
    return transform_to_schema_single_pass(df, source_file)


def save_silver_data(df: pd.DataFrame, output_path: Path, use_compression: bool = True):
    """
    Save cleaned data to Silver layer with optimized writing.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save cleaned CSV
        use_compression: If True, use gzip compression for faster I/O
    """
    try:
        logger.info(f"Saving cleaned data to: {output_path}")
        
        if use_compression and len(df) > 10000:
            # Use compressed CSV for large files (faster I/O, smaller files)
            output_path_compressed = output_path.with_suffix('.csv.gz')
            df.to_csv(
                output_path_compressed,
                index=False,
                compression='gzip',
            )
            file_size = output_path_compressed.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Saved {len(df):,} rows ({file_size:.2f} MB compressed)")
            # Also save uncompressed version for compatibility
            df.to_csv(output_path, index=False)
        else:
            # Standard CSV for smaller files
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
    Process a single Bronze CSV file through Silver layer (optimized).
    
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
    
    # Step 1: Remove duplicates (optimized)
    logger.info("Step 1: Removing duplicates...")
    df = remove_duplicates_fast(df)
    
    # Step 2: Build column mapping once (optimization)
    col_map = build_column_mapping(df)
    
    # Step 3: Check if we have coordinates
    has_coordinates = col_map.get('lat') and col_map.get('lon')
    
    if not has_coordinates:
        logger.info(f"Dataset {csv_path.name} does not have coordinate columns.")
        logger.info("This appears to be an aggregate/statistical dataset - processing with flexible schema.")
    
    # Step 4: Transform to schema (flexible - works with or without coordinates)
    logger.info("Step 4: Transforming to flexible schema...")
    try:
        if has_coordinates:
            # Use standard transformation for datasets with coordinates
            df_clean = transform_to_schema_single_pass(df, source_file)
        else:
            # Use flexible transformation for datasets without coordinates
            df_clean = transform_to_schema_flexible(df, source_file)
    except Exception as e:
        logger.error(f"Transformation failed for {csv_path.name}: {e}")
        return False
    
    # Step 5: Save cleaned data (optimized)
    if output_file:
        output_path = Path(output_file)
    else:
        # Generate output filename based on source file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract dataset name from source file
        dataset_name = csv_path.stem.split("_")[0] if "_" in csv_path.stem else csv_path.stem
        output_path = silver_dir / f"cleaned_{dataset_name}_{timestamp}.csv"
    
    try:
        save_silver_data(df_clean, output_path, use_compression=True)
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
        csv_files = find_all_bronze_files(raw_dir, only_recent=True)
        if not csv_files:
            logger.error(f"No recent CSV files found in {raw_dir} (last 24 hours)")
            logger.info("Tip: If you want to process older files, modify find_all_bronze_files() or use --input to specify files")
            sys.exit(1)
        
        logger.info(f"Processing {len(csv_files)} Bronze files in parallel...")
        logger.info("")
        
        # Process files in parallel (optimization)
        max_workers = min(4, len(csv_files))  # Use up to 4 workers
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, csv_path, skip_validation=skip_validation): csv_path
                for csv_path in csv_files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                csv_path = futures[future]
                try:
                    success = future.result()
                    results[csv_path] = success
                    if success:
                        logger.success(f"✓ [{i}/{len(csv_files)}] Processed: {csv_path.name}")
                    else:
                        logger.error(f"✗ [{i}/{len(csv_files)}] Failed: {csv_path.name}")
                except Exception as e:
                    logger.error(f"✗ [{i}/{len(csv_files)}] Exception processing {csv_path.name}: {e}")
                    results[csv_path] = False
        
        success_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - success_count
        
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

