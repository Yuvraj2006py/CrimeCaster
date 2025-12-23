"""
Local data loader for ML training.

Loads crime data from local Parquet files instead of database.
This allows unlimited data storage and faster training.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from loguru import logger

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("pyarrow not installed. Install with: pip install pyarrow")


def load_training_data(
    data_dir: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dataset_types: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load crime data from local Parquet files for ML training.
    
    Parquet files are stored in data/gold/parquet/ after Gold layer processing.
    This function loads and combines all matching Parquet files.
    
    Args:
        data_dir: Project root directory (e.g., Path(".") or Path(__file__).parent.parent)
        start_date: Only load records from this date onwards (inclusive)
        end_date: Only load records up to this date (inclusive)
        dataset_types: Filter by dataset types (e.g., ['major-crime-indicators'])
        max_files: Maximum number of files to load (for testing, None = all)
        
    Returns:
        Combined DataFrame with all matching records
        
    Raises:
        ValueError: If Parquet directory not found or no files found
        ImportError: If pyarrow not installed
    """
    if not HAS_PYARROW:
        raise ImportError("pyarrow is required for loading Parquet files. Install with: pip install pyarrow")
    
    # Resolve data directory
    data_dir = Path(data_dir).resolve()
    parquet_dir = data_dir / "data" / "gold" / "parquet"
    
    if not parquet_dir.exists():
        raise ValueError(
            f"Parquet directory not found: {parquet_dir}\n"
            f"Run the Gold layer first to generate Parquet files."
        )
    
    # Find all Parquet files
    parquet_files = sorted(list(parquet_dir.glob("*.parquet")))
    
    if not parquet_files:
        raise ValueError(
            f"No Parquet files found in {parquet_dir}\n"
            f"Run the Gold layer first to generate Parquet files."
        )
    
    if max_files:
        parquet_files = parquet_files[:max_files]
        logger.info(f"Limiting to first {max_files} files (for testing)")
    
    logger.info(f"Found {len(parquet_files)} Parquet files in {parquet_dir}")
    
    # Load all files
    dfs = []
    total_rows = 0
    skipped_files = 0
    
    for parquet_file in parquet_files:
        try:
            # Load file
            df = pd.read_parquet(parquet_file)
            
            if len(df) == 0:
                logger.debug(f"  Skipping empty file: {parquet_file.name}")
                skipped_files += 1
                continue
            
            # Apply filters
            initial_count = len(df)
            
            if start_date:
                if 'occurred_at' in df.columns:
                    df = df[df['occurred_at'] >= pd.Timestamp(start_date)]
                else:
                    logger.warning(f"  No 'occurred_at' column in {parquet_file.name}, skipping date filter")
            
            if end_date:
                if 'occurred_at' in df.columns:
                    df = df[df['occurred_at'] <= pd.Timestamp(end_date)]
                else:
                    logger.warning(f"  No 'occurred_at' column in {parquet_file.name}, skipping date filter")
            
            if dataset_types:
                if 'dataset_type' in df.columns:
                    df = df[df['dataset_type'].isin(dataset_types)]
                else:
                    logger.warning(f"  No 'dataset_type' column in {parquet_file.name}, skipping dataset filter")
            
            if len(df) == 0:
                logger.debug(f"  No matching records in {parquet_file.name} after filtering")
                skipped_files += 1
                continue
            
            dfs.append(df)
            total_rows += len(df)
            
            if initial_count != len(df):
                logger.info(
                    f"  Loaded {len(df):,}/{initial_count:,} rows from {parquet_file.name} "
                    f"(filtered: {initial_count - len(df):,})"
                )
            else:
                logger.info(f"  Loaded {len(df):,} rows from {parquet_file.name}")
            
        except Exception as e:
            logger.warning(f"  Error loading {parquet_file.name}: {e}")
            skipped_files += 1
            continue
    
    if not dfs:
        raise ValueError(
            f"No data loaded from Parquet files.\n"
            f"  Files found: {len(parquet_files)}\n"
            f"  Files skipped: {skipped_files}\n"
            f"  Check your date/dataset filters if applicable."
        )
    
    # Combine all DataFrames
    logger.info(f"Combining {len(dfs)} DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info("=" * 60)
    logger.info(f"✅ Loaded {len(combined_df):,} total records for training")
    logger.info(f"   From {len(dfs)} files ({skipped_files} skipped)")
    if start_date or end_date:
        logger.info(f"   Date range: {start_date or 'any'} to {end_date or 'any'}")
    if dataset_types:
        logger.info(f"   Dataset types: {', '.join(dataset_types)}")
    logger.info("=" * 60)
    
    return combined_df


def get_data_statistics(data_dir: Path) -> Dict:
    """
    Get statistics about available training data without loading all files.
    
    This is a lightweight operation that reads Parquet metadata only.
    
    Args:
        data_dir: Project root directory
        
    Returns:
        Dict with data statistics:
        - total_files: Number of Parquet files
        - total_rows: Total number of rows (from metadata)
        - total_size_mb: Total file size in MB
        - date_range: Dict with 'min' and 'max' dates (if available)
        - sample_columns: List of column names from first file
    """
    if not HAS_PYARROW:
        return {"error": "pyarrow not installed"}
    
    data_dir = Path(data_dir).resolve()
    parquet_dir = data_dir / "data" / "gold" / "parquet"
    
    if not parquet_dir.exists():
        return {"error": f"Parquet directory not found: {parquet_dir}"}
    
    parquet_files = sorted(list(parquet_dir.glob("*.parquet")))
    
    if not parquet_files:
        return {"error": f"No Parquet files found in {parquet_dir}"}
    
    total_rows = 0
    total_size_mb = 0
    date_range = None
    sample_columns = None
    
    for parquet_file in parquet_files:
        try:
            # Get file size
            total_size_mb += parquet_file.stat().st_size / (1024 * 1024)
            
            # Read metadata without loading full file
            parquet_file_obj = pq.ParquetFile(parquet_file)
            total_rows += parquet_file_obj.metadata.num_rows
            
            # Get sample columns and date range from first file
            if sample_columns is None:
                # Read first 1000 rows for sampling (Parquet doesn't support nrows, so read and slice)
                df_sample = pd.read_parquet(parquet_file).head(1000)
                sample_columns = list(df_sample.columns)
                
                if 'occurred_at' in df_sample.columns:
                    # Get full date range by reading all files (lightweight)
                    # For efficiency, we'll sample the first and last files
                    if len(parquet_files) == 1:
                        # Single file - get full range from sample
                        date_range = {
                            'min': df_sample['occurred_at'].min(),
                            'max': df_sample['occurred_at'].max(),
                        }
                    else:
                        # Multiple files - sample first and last (read first 1000 rows each)
                        df_first = pd.read_parquet(parquet_files[0]).head(1000)
                        df_last = pd.read_parquet(parquet_files[-1]).head(1000)
                        date_range = {
                            'min': min(
                                df_first['occurred_at'].min() if 'occurred_at' in df_first.columns else None,
                                df_sample['occurred_at'].min() if 'occurred_at' in df_sample.columns else None,
                            ),
                            'max': max(
                                df_last['occurred_at'].max() if 'occurred_at' in df_last.columns else None,
                                df_sample['occurred_at'].max() if 'occurred_at' in df_sample.columns else None,
                            ),
                        }
        except Exception as e:
            logger.warning(f"Error reading {parquet_file.name}: {e}")
            continue
    
    return {
        "total_files": len(parquet_files),
        "total_rows": total_rows,
        "total_size_mb": round(total_size_mb, 2),
        "date_range": date_range,
        "sample_columns": sample_columns,
    }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    # Get statistics
    print("=" * 60)
    print("Training Data Statistics")
    print("=" * 60)
    stats = get_data_statistics(project_root)
    if "error" in stats:
        print(f"Error: {stats['error']}")
    else:
        print(f"Total files: {stats['total_files']}")
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Total size: {stats['total_size_mb']} MB")
        if stats['date_range']:
            print(f"Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
        print(f"Columns: {', '.join(stats['sample_columns'][:10])}...")
    
    # Load sample data
    print("\n" + "=" * 60)
    print("Loading Sample Data (last 2 years)")
    print("=" * 60)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        df = load_training_data(
            data_dir=project_root,
            start_date=start_date,
            end_date=end_date,
            max_files=5,  # Limit for testing
        )
        
        print(f"\nLoaded DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error loading data: {e}")

