"""
Feature Engineering for Crime Caster ML Training.

Creates spatiotemporal features from processed crime data for ML model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from loguru import logger
import h3

from training.data_loader import load_training_data

# Feature engineering constants
H3_RESOLUTION = 9  # Match Gold layer resolution
TORONTO_LAT_MIN = 43.5
TORONTO_LAT_MAX = 43.9
TORONTO_LON_MIN = -79.6
TORONTO_LON_MAX = -79.1

# Time windows for historical features (in hours)
TIME_WINDOWS = {
    "1h": 1,
    "24h": 24,
    "7d": 24 * 7,
    "30d": 24 * 30,
}


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from occurred_at column.
    
    Features:
    - hour_of_day (0-23)
    - day_of_week (0-6, Monday=0)
    - is_weekend (boolean)
    - is_night (boolean, 10pm-6am)
    - month (1-12)
    - season (1-4: Spring=1, Summer=2, Fall=3, Winter=4)
    - day_of_month (1-31)
    - week_of_year (1-52)
    
    Args:
        df: DataFrame with 'occurred_at' column
        
    Returns:
        DataFrame with temporal features added
    """
    if 'occurred_at' not in df.columns:
        logger.warning("No 'occurred_at' column found. Skipping temporal features.")
        return df
    
    logger.info("Extracting temporal features...")
    
    # Ensure occurred_at is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['occurred_at']):
        df['occurred_at'] = pd.to_datetime(df['occurred_at'], errors='coerce')
    
    # Remove rows with invalid dates
    initial_count = len(df)
    df = df[df['occurred_at'].notna()].copy()
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df):,} rows with invalid dates")
    
    # Extract temporal features
    df['hour_of_day'] = df['occurred_at'].dt.hour
    df['day_of_week'] = df['occurred_at'].dt.dayofweek  # Monday=0
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
    df['is_night'] = (df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)
    df['month'] = df['occurred_at'].dt.month
    df['day_of_month'] = df['occurred_at'].dt.day
    df['week_of_year'] = df['occurred_at'].dt.isocalendar().week
    
    # Season: Spring (Mar-May=1), Summer (Jun-Aug=2), Fall (Sep-Nov=3), Winter (Dec-Feb=4)
    df['season'] = df['month'].apply(
        lambda m: 1 if 3 <= m <= 5 else (2 if 6 <= m <= 8 else (3 if 9 <= m <= 11 else 4))
    )
    
    # Holiday detection (simplified - can be enhanced with holiday calendar)
    # For now, mark major holidays
    df['is_holiday'] = False  # Placeholder - can add holiday detection later
    
    logger.info(f"Extracted temporal features for {len(df):,} records")
    
    return df


def create_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time window columns for rolling aggregations.
    
    Creates rounded time windows (1-hour bins) for grouping historical features.
    
    Args:
        df: DataFrame with 'occurred_at' column
        
    Returns:
        DataFrame with 'time_window' column added
    """
    if 'occurred_at' not in df.columns:
        return df
    
    # Round to nearest hour for time windows
    df['time_window'] = df['occurred_at'].dt.floor('h')  # Use 'h' instead of deprecated 'H'
    
    return df


def compute_historical_features(
    df: pd.DataFrame,
    group_by: str = 'h3_index',
    time_col: str = 'occurred_at'
) -> pd.DataFrame:
    """
    Compute historical crime count features for each location/time (optimized).
    
    Features:
    - crimes_last_1h: Count in same H3 cell in last 1 hour
    - crimes_last_24h: Count in same H3 cell in last 24 hours
    - crimes_last_7d: Count in same H3 cell in last 7 days
    - rolling_avg_30d: Rolling average over 30 days
    - same_hour_last_week: Count at same hour, same day of week, 1 week ago
    
    Args:
        df: DataFrame with crime records
        group_by: Column to group by (usually 'h3_index')
        time_col: Time column name
        
    Returns:
        DataFrame with historical features added
    """
    if group_by not in df.columns or time_col not in df.columns:
        logger.warning(f"Missing required columns ({group_by}, {time_col}). Skipping historical features.")
        return df
    
    logger.info(f"Computing historical features grouped by {group_by} (optimized)...")
    
    # Sort by time for rolling calculations
    df = df.sort_values([group_by, time_col]).copy()
    
    # Initialize feature columns
    df['crimes_last_1h'] = 0
    df['crimes_last_24h'] = 0
    df['crimes_last_7d'] = 0
    df['rolling_avg_30d'] = 0.0
    df['same_hour_last_week'] = 0
    
    # Use vectorized operations with groupby and rolling windows
    # This is much faster than row-by-row iteration
    
    # Group by location
    grouped = df.groupby(group_by)
    
    # For each group, compute rolling features
    result_dfs = []
    
    for h3_idx, group in grouped:
        if len(group) == 0:
            continue
        
        # Sort by time
        group = group.sort_values(time_col).copy()
        group_times = pd.to_datetime(group[time_col])
        
        # Use numpy for faster comparisons
        times_array = group_times.values
        
        # Initialize arrays for features
        crimes_1h = np.zeros(len(group), dtype=int)
        crimes_24h = np.zeros(len(group), dtype=int)
        crimes_7d = np.zeros(len(group), dtype=int)
        rolling_30d = np.zeros(len(group), dtype=float)
        same_hour_week = np.zeros(len(group), dtype=int)
        
        # For each record, count previous crimes in time windows
        for i in range(len(group)):
            current_time = times_array[i]
            
            # Time deltas
            time_1h = current_time - np.timedelta64(1, 'h')
            time_24h = current_time - np.timedelta64(24, 'h')
            time_7d = current_time - np.timedelta64(7, 'D')
            time_30d = current_time - np.timedelta64(30, 'D')
            time_week_ago = current_time - np.timedelta64(7, 'D')
            time_week_ago_1h = time_week_ago - np.timedelta64(1, 'h')
            
            # Count crimes in each window (excluding current record)
            mask_1h = (times_array < current_time) & (times_array >= time_1h)
            crimes_1h[i] = mask_1h.sum()
            
            mask_24h = (times_array < current_time) & (times_array >= time_24h)
            crimes_24h[i] = mask_24h.sum()
            
            mask_7d = (times_array < current_time) & (times_array >= time_7d)
            crimes_7d[i] = mask_7d.sum()
            
            mask_30d = (times_array < current_time) & (times_array >= time_30d)
            if mask_30d.sum() > 0:
                rolling_30d[i] = mask_30d.sum() / 30.0
            
            # Same hour last week (within 1 hour window)
            mask_same_hour = (times_array >= time_week_ago_1h) & (times_array < time_week_ago)
            same_hour_week[i] = mask_same_hour.sum()
        
        # Assign computed features
        group['crimes_last_1h'] = crimes_1h
        group['crimes_last_24h'] = crimes_24h
        group['crimes_last_7d'] = crimes_7d
        group['rolling_avg_30d'] = rolling_30d
        group['same_hour_last_week'] = same_hour_week
        
        result_dfs.append(group)
    
    # Combine all groups
    if result_dfs:
        df = pd.concat(result_dfs, ignore_index=True)
    
    logger.info(f"Computed historical features for {len(df):,} records")
    
    return df


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spatial features using H3 hexagons (optimized).
    
    Features:
    - h3_neighbors_count: Number of crimes in neighboring H3 cells (last 24h)
    - h3_ring1_crimes: Crimes in ring-1 neighbors (6 cells)
    - h3_ring2_crimes: Crimes in ring-2 neighbors (12 cells)
    
    Args:
        df: DataFrame with 'h3_index' column
        
    Returns:
        DataFrame with spatial features added
    """
    if 'h3_index' not in df.columns:
        logger.warning("No 'h3_index' column found. Skipping spatial features.")
        return df
    
    logger.info("Computing spatial features (optimized)...")
    
    # Remove rows without H3 index
    initial_count = len(df)
    df = df[df['h3_index'].notna()].copy()
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df):,} rows without H3 index")
    
    # Initialize feature columns
    df['h3_neighbors_count'] = 0
    df['h3_ring1_crimes'] = 0
    df['h3_ring2_crimes'] = 0
    
    # Group by time window (1-hour bins) for neighbor calculations
    if 'time_window' not in df.columns:
        df = create_time_windows(df)
    
    # Pre-compute neighbor mappings for all unique H3 indices
    unique_h3 = df['h3_index'].unique()
    neighbor_cache = {}
    
    logger.info(f"Pre-computing neighbor mappings for {len(unique_h3):,} unique H3 cells...")
    for h3_idx in unique_h3:
        if pd.isna(h3_idx):
            continue
        try:
            ring1 = set(h3.grid_ring(h3_idx, 1))
            ring2 = set(h3.grid_ring(h3_idx, 2))
            neighbor_cache[h3_idx] = {'ring1': ring1, 'ring2': ring2}
        except Exception as e:
            logger.debug(f"Error computing neighbors for H3 {h3_idx}: {e}")
            neighbor_cache[h3_idx] = {'ring1': set(), 'ring2': set()}
    
    # For each time window, compute neighbor features
    result_dfs = []
    time_windows = df['time_window'].unique()
    
    for time_win in time_windows:
        time_group = df[df['time_window'] == time_win].copy()
        
        # Create mapping of H3 index to crime count in this time window
        h3_counts = time_group['h3_index'].value_counts().to_dict()
        
        # Compute neighbor features for all H3 indices in this time window
        ring1_crimes_dict = {}
        ring2_crimes_dict = {}
        neighbors_count_dict = {}
        
        for h3_idx in time_group['h3_index'].unique():
            if pd.isna(h3_idx) or h3_idx not in neighbor_cache:
                continue
            
            neighbors = neighbor_cache[h3_idx]
            
            # Count crimes in ring-1 neighbors
            ring1_crimes = sum(h3_counts.get(neighbor, 0) for neighbor in neighbors['ring1'])
            ring1_crimes_dict[h3_idx] = ring1_crimes
            
            # Count crimes in ring-2 neighbors
            ring2_crimes = sum(h3_counts.get(neighbor, 0) for neighbor in neighbors['ring2'])
            ring2_crimes_dict[h3_idx] = ring2_crimes
            
            # Total neighbors
            neighbors_count_dict[h3_idx] = ring1_crimes + ring2_crimes
        
        # Map features to DataFrame
        time_group['h3_ring1_crimes'] = time_group['h3_index'].map(ring1_crimes_dict).fillna(0).astype(int)
        time_group['h3_ring2_crimes'] = time_group['h3_index'].map(ring2_crimes_dict).fillna(0).astype(int)
        time_group['h3_neighbors_count'] = time_group['h3_index'].map(neighbors_count_dict).fillna(0).astype(int)
        
        result_dfs.append(time_group)
    
    # Combine all time windows
    if result_dfs:
        df = pd.concat(result_dfs, ignore_index=True)
    
    # Step: Compute distance to hotspots
    logger.info("Computing distance to hotspots...")
    df = compute_distance_to_hotspots(df)
    
    logger.info(f"Computed spatial features for {len(df):,} records")
    
    return df


def compute_distance_to_hotspots(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute distance to high-crime hotspots.
    
    Identifies top N high-crime H3 cells and computes distance from each record.
    
    Args:
        df: DataFrame with 'h3_index' and 'latitude'/'longitude' columns
        top_n: Number of hotspots to identify
        
    Returns:
        DataFrame with 'distance_to_hotspot' column added
    """
    if 'h3_index' not in df.columns or 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.warning("Missing required columns for hotspot distance. Skipping.")
        return df
    
    logger.info(f"Identifying top {top_n} crime hotspots...")
    
    # Identify hotspots: H3 cells with most crimes
    h3_crime_counts = df['h3_index'].value_counts()
    hotspot_h3 = h3_crime_counts.head(top_n).index.tolist()
    
    # Get centroids of hotspot H3 cells
    hotspot_coords = []
    for h3_idx in hotspot_h3:
        try:
            # Get centroid of H3 cell
            lat, lon = h3.cell_to_latlng(h3_idx)
            hotspot_coords.append((lat, lon))
        except Exception as e:
            logger.debug(f"Error getting centroid for H3 {h3_idx}: {e}")
            continue
    
    if not hotspot_coords:
        logger.warning("No valid hotspot coordinates. Setting distance to 0.")
        df['distance_to_hotspot'] = 0.0
        return df
    
    # Convert to numpy arrays for vectorized distance calculation
    hotspot_lats = np.array([c[0] for c in hotspot_coords])
    hotspot_lons = np.array([c[1] for c in hotspot_coords])
    
    # Compute distance from each record to nearest hotspot
    # Using Haversine distance (approximate, good for small distances)
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in km."""
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    # Vectorized distance calculation
    record_lats = df['latitude'].values
    record_lons = df['longitude'].values
    
    # Compute distance to all hotspots, then take minimum
    min_distances = np.full(len(df), np.inf)
    
    for i, (hot_lat, hot_lon) in enumerate(hotspot_coords):
        distances = haversine_distance(record_lats, record_lons, hot_lat, hot_lon)
        min_distances = np.minimum(min_distances, distances)
    
    df['distance_to_hotspot'] = min_distances
    
    logger.info(f"Computed distance to hotspots for {len(df):,} records")
    logger.info(f"  Average distance: {df['distance_to_hotspot'].mean():.2f} km")
    logger.info(f"  Min distance: {df['distance_to_hotspot'].min():.2f} km")
    logger.info(f"  Max distance: {df['distance_to_hotspot'].max():.2f} km")
    
    return df


def compute_aggregate_features(
    df: pd.DataFrame,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compute features from aggregate datasets (neighborhood rates, annual trends).
    
    Features:
    - neighborhood_crime_rate: Crime rate from neighbourhood-crime-rates dataset
    - annual_crime_trend: Trend from police-annual-statistical-report datasets
    
    Args:
        df: Main DataFrame with crime records
        data_dir: Project root directory for loading aggregate datasets
        
    Returns:
        DataFrame with aggregate features added
    """
    logger.info("Computing aggregate dataset features...")
    
    # Initialize feature columns
    df['neighborhood_crime_rate'] = 0.0
    df['annual_crime_trend'] = 0.0
    
    if data_dir is None:
        logger.warning("No data_dir provided. Skipping aggregate features.")
        return df
    
    # Try to load neighbourhood-crime-rates dataset
    try:
        parquet_dir = Path(data_dir) / "data" / "gold" / "parquet"
        neighbourhood_files = list(parquet_dir.glob("neighbourhood-crime-rates*.parquet"))
        
        if neighbourhood_files and 'neighbourhood' in df.columns:
            logger.info(f"Loading neighbourhood crime rates from {len(neighbourhood_files)} files...")
            agg_dfs = []
            for file in neighbourhood_files:
                try:
                    agg_df = pd.read_parquet(file)
                    # Look for columns that might contain crime rates
                    # This is dataset-specific, so we'll use a flexible approach
                    if len(agg_df) > 0:
                        agg_dfs.append(agg_df)
                except Exception as e:
                    logger.debug(f"Error loading {file.name}: {e}")
                    continue
            
            if agg_dfs:
                # Combine aggregate data
                combined_agg = pd.concat(agg_dfs, ignore_index=True)
                
                # Try to extract neighborhood rates
                # The structure depends on the dataset, so we'll compute a simple rate
                if 'neighbourhood' in combined_agg.columns:
                    # Count crimes per neighborhood in aggregate data
                    neighbourhood_counts = combined_agg['neighbourhood'].value_counts().to_dict()
                    
                    # Map to main DataFrame
                    df['neighborhood_crime_rate'] = df['neighbourhood'].map(
                        lambda x: neighbourhood_counts.get(x, 0.0) if pd.notna(x) else 0.0
                    ).astype(float)
                    
                    logger.info(f"Mapped neighborhood crime rates for {df['neighborhood_crime_rate'].gt(0).sum():,} records")
        
    except Exception as e:
        logger.warning(f"Error loading aggregate datasets: {e}")
    
    # Annual trend: compute from historical data in main DataFrame
    if 'occurred_at' in df.columns and 'month' in df.columns:
        # Compute year-over-year trend
        df['year'] = pd.to_datetime(df['occurred_at']).dt.year
        
        # Group by year and compute average crimes per month
        yearly_avg = df.groupby('year').size() / 12.0  # Approximate monthly average
        
        if len(yearly_avg) > 1:
            # Compute trend: (current_year - previous_year) / previous_year
            years = sorted(yearly_avg.index)
            trend_dict = {}
            
            for i, year in enumerate(years):
                if i == 0:
                    trend = 0.0
                else:
                    prev_avg = yearly_avg[years[i-1]]
                    curr_avg = yearly_avg[year]
                    if prev_avg > 0:
                        trend = (curr_avg - prev_avg) / prev_avg
                    else:
                        trend = 0.0
                
                trend_dict[year] = trend
            
            df['annual_crime_trend'] = df['year'].map(trend_dict).fillna(0.0)
            logger.info(f"Computed annual trends for {len(yearly_avg)} years")
    
    logger.info(f"Computed aggregate features for {len(df):,} records")
    
    return df


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for ML training.
    
    Targets:
    - target_binary: 1 if crime occurred, 0 otherwise (always 1 for training data)
    - target_count: Number of crimes (always 1 for individual records)
    
    For training, we'll create time-windowed targets:
    - For each H3 cell + time window, count crimes
    - Binary: 1 if count > 0, 0 otherwise
    - Count: actual count
    
    Args:
        df: DataFrame with crime records
        
    Returns:
        DataFrame with target variables added
    """
    logger.info("Creating target variables...")
    
    # For individual crime records, target is always 1
    df['target_binary'] = 1
    df['target_count'] = 1
    
    # Create aggregated targets by H3 + time window
    if 'h3_index' in df.columns and 'time_window' in df.columns:
        # Group by H3 and time window to get counts
        grouped = df.groupby(['h3_index', 'time_window']).size().reset_index(name='crime_count')
        
        # Merge back to original DataFrame
        df = df.merge(
            grouped[['h3_index', 'time_window', 'crime_count']],
            on=['h3_index', 'time_window'],
            how='left'
        )
        
        # Update target_count to be the count for this time window
        df['target_count'] = df['crime_count'].fillna(1)
        
        # Binary target: 1 if count > 0 (always true for training data, but useful for inference)
        df['target_binary'] = (df['target_count'] > 0).astype(int)
        
        # Drop intermediate column
        df = df.drop(columns=['crime_count'])
    
    logger.info(f"Created target variables for {len(df):,} records")
    
    return df


def engineer_features(
    data_dir: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dataset_types: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    output_path: Optional[Path] = None,
    include_aggregate: bool = False,
) -> pd.DataFrame:
    """
    Main function to engineer all features from Parquet data.
    
    Args:
        data_dir: Project root directory
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
        dataset_types: Filter by dataset types (optional)
        max_files: Limit number of files (for testing)
        output_path: Path to save engineered features (optional)
        include_aggregate: Whether to include aggregate dataset features
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Step 1: Load data from Parquet files
    logger.info("Step 1: Loading data from Parquet files...")
    df = load_training_data(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        dataset_types=dataset_types,
        max_files=max_files,
    )
    
    # Filter to datasets with coordinates (for now, skip aggregate datasets)
    # Handle both boolean and string types for has_coordinates column
    if 'has_coordinates' in df.columns:
        initial_count = len(df)
        
        # Convert string "true"/"false" to boolean if needed, or handle boolean directly
        if df['has_coordinates'].dtype == 'object':
            # Handle string values (case-insensitive)
            df_filtered = df[df['has_coordinates'].astype(str).str.lower() == 'true'].copy()
        else:
            # Handle boolean values
            df_filtered = df[df['has_coordinates'] == True].copy()
        
        # If has_coordinates filter resulted in 0 records, fall back to h3_index
        if len(df_filtered) == 0 and 'h3_index' in df.columns:
            logger.warning("has_coordinates filter resulted in 0 records, falling back to h3_index filter")
            initial_count = len(df)
            df_filtered = df[df['h3_index'].notna()].copy()
            if len(df_filtered) < initial_count:
                logger.info(f"Filtered to {len(df_filtered):,} records with H3 index (removed {initial_count - len(df_filtered):,} records)")
        elif len(df_filtered) < initial_count:
            logger.info(f"Filtered to {len(df_filtered):,} records with coordinates (removed {initial_count - len(df_filtered):,} aggregate records)")
        
        df = df_filtered
    elif 'h3_index' in df.columns:
        # Filter to records with H3 index
        initial_count = len(df)
        df = df[df['h3_index'].notna()].copy()
        if len(df) < initial_count:
            logger.info(f"Filtered to {len(df):,} records with H3 index (removed {initial_count - len(df):,} records)")
    
    if len(df) == 0:
        raise ValueError("No data with coordinates found. Cannot create features.")
    
    # Step 2: Extract temporal features
    logger.info("\nStep 2: Extracting temporal features...")
    df = extract_temporal_features(df)
    
    # Step 3: Create time windows
    logger.info("\nStep 3: Creating time windows...")
    df = create_time_windows(df)
    
    # Step 4: Compute historical features
    logger.info("\nStep 4: Computing historical features...")
    df = compute_historical_features(df, group_by='h3_index', time_col='occurred_at')
    
    # Step 5: Compute spatial features
    logger.info("\nStep 5: Computing spatial features...")
    df = compute_spatial_features(df)
    
    # Step 6: Compute aggregate features (if requested)
    if include_aggregate:
        logger.info("\nStep 6: Computing aggregate features...")
        df = compute_aggregate_features(df, data_dir=data_dir)
    
    # Step 7: Create target variables
    logger.info("\nStep 7: Creating target variables...")
    df = create_target_variables(df)
    
    # Step 8: Save to Parquet if output path provided
    if output_path:
        logger.info(f"\nStep 8: Saving engineered features to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression='snappy', index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved {len(df):,} records to {output_path} ({file_size_mb:.2f} MB)")
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Total features: {len(df.columns)}")
    logger.info(f"Feature columns: {', '.join(df.columns.tolist())}")
    
    return df


def main():
    """Main entry point for feature engineering."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Engineer features for ML training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        help="Filter by dataset types (e.g., major-crime-indicators)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of Parquet files (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for engineered features Parquet file"
    )
    parser.add_argument(
        "--include-aggregate",
        action="store_true",
        help="Include aggregate dataset features"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Default output path
    output_path = args.output
    if not output_path:
        data_dir = Path(args.data_dir)
        output_dir = data_dir / "data" / "features"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"engineered_features_{timestamp}.parquet"
    
    # Run feature engineering
    df = engineer_features(
        data_dir=Path(args.data_dir),
        start_date=start_date,
        end_date=end_date,
        dataset_types=args.dataset_types,
        max_files=args.max_files,
        output_path=output_path,
        include_aggregate=args.include_aggregate,
    )
    
    logger.info(f"\n✅ Feature engineering complete!")
    logger.info(f"   Output saved to: {output_path}")
    logger.info(f"   Records: {len(df):,}")
    logger.info(f"   Features: {len(df.columns)}")


if __name__ == "__main__":
    main()

