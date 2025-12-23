"""
Bronze layer: Raw CSV ingestion from Toronto Open Data Portal.

Downloads and loads crime data CSV files, performs basic validation,
and tracks ingestion metadata. Supports multiple datasets with parallel downloads.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import httpx
from loguru import logger
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from validation.schema_validator import validate_csv
from ingestion.bronze.dataset_config import (
    TORONTO_DATASETS,
    TORONTO_CKAN_API,
    get_all_dataset_keys,
    get_dataset_info,
    get_dataset_url,
)

# Try to import chardet for encoding detection, fallback to manual detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logger.warning("chardet not installed. Encoding detection will use fallback method.")

# Load environment variables
load_dotenv()

# Legacy: Toronto Open Data Portal URL for Major Crime Indicators (for backward compatibility)
TORONTO_CRIME_DATA_URL = (
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
    "major-crime-indicators/resource/6f0a0e0e-8c0a-4b3a-9f3a-1e8b3a4f5e6b/download/"
    "major-crime-indicators.csv"
)

# Expected CSV columns (Toronto Open Data format may vary)
EXPECTED_COLUMNS = [
    "event_unique_id",
    "occurrence_date",
    "occurrence_time",
    "report_date",
    "report_time",
    "major_crime_indicator",
    "subtype",
    "mci_category",
    "division",
    "location_type",
    "premises_type",
    "hood_id",
    "neighbourhood",
    "lat",
    "long",
]


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def get_data_directory() -> Path:
    """Get or create data/raw directory."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def discover_resource_id(dataset_key: str, timeout: int = 30) -> Optional[str]:
    """
    Discover resource ID and download URL for a dataset using CKAN API.
    
    Uses proper CKAN API pattern:
    - For datastore_active resources: Use /datastore/dump/ endpoint
    - For non-datastore resources: Use resource_show to get download URL
    
    Args:
        dataset_key: Dataset key from TORONTO_DATASETS
        timeout: Request timeout in seconds
        
    Returns:
        Resource ID (string) for datastore resources, or direct URL (string) for file resources, or None if not found
    """
    try:
        # Step 1: Get package metadata
        api_url = f"{TORONTO_CKAN_API}/package_show"
        params = {"id": dataset_key}
        
        logger.debug(f"Discovering resource for {dataset_key} via CKAN API...")
        
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                logger.warning(f"CKAN API returned success=False for {dataset_key}")
                if "error" in data:
                    logger.warning(f"API error: {data['error']}")
                return None
            
            if "result" not in data:
                logger.warning(f"No result in CKAN API response for {dataset_key}")
                return None
            
            resources = data["result"].get("resources", [])
            
            if not resources:
                logger.warning(f"No resources found for dataset {dataset_key}")
                return None
            
            # Step 2: Process each resource
            # Prefer CSV format, but accept any downloadable resource
            csv_resource = None
            for resource in resources:
                resource_format = resource.get("format", "").upper()
                if resource_format == "CSV":
                    csv_resource = resource
                    break
            
            # If no CSV found, try to find any downloadable resource
            if not csv_resource and resources:
                for resource in resources:
                    if resource.get("url") or resource.get("download_url") or resource.get("datastore_active"):
                        csv_resource = resource
                        break
            
            if not csv_resource:
                logger.warning(f"No downloadable resource found for {dataset_key}")
                logger.debug(f"Available resources: {[r.get('format') for r in resources]}")
                return None
            
            resource_id = csv_resource.get("id")
            is_datastore = csv_resource.get("datastore_active", False)
            
            # Step 3: Handle datastore_active resources
            if is_datastore and resource_id:
                # For datastore resources, return resource ID
                # Download will use /datastore/dump/ endpoint
                logger.info(f"Discovered datastore resource ID for {dataset_key}: {resource_id}")
                TORONTO_DATASETS[dataset_key]["resource_id"] = resource_id
                TORONTO_DATASETS[dataset_key]["datastore_active"] = True
                return resource_id
            
            # Step 4: Handle non-datastore resources - get download URL
            if not is_datastore and resource_id:
                # Get resource metadata to find download URL
                resource_show_url = f"{TORONTO_CKAN_API}/resource_show"
                resource_params = {"id": resource_id}
                
                try:
                    resource_response = client.get(resource_show_url, params=resource_params)
                    resource_response.raise_for_status()
                    resource_data = resource_response.json()
                    
                    if resource_data.get("success") and "result" in resource_data:
                        resource_url = (
                            resource_data["result"].get("url") or 
                            resource_data["result"].get("download_url")
                        )
                        
                        if resource_url:
                            logger.info(f"Found download URL for {dataset_key}: {resource_url}")
                            TORONTO_DATASETS[dataset_key]["direct_url"] = resource_url
                            TORONTO_DATASETS[dataset_key]["resource_id"] = resource_id
                            return resource_url
                except Exception as e:
                    logger.warning(f"Failed to get resource metadata for {resource_id}: {e}")
            
            # Fallback: Use URL from package_show if available
            resource_url = csv_resource.get("url") or csv_resource.get("download_url")
            if resource_url:
                logger.info(f"Found direct URL for {dataset_key}: {resource_url}")
                TORONTO_DATASETS[dataset_key]["direct_url"] = resource_url
                if resource_id:
                    TORONTO_DATASETS[dataset_key]["resource_id"] = resource_id
                return resource_url
            
            # If we have resource_id but no URL, return it (will construct URL)
            if resource_id:
                logger.info(f"Discovered resource ID for {dataset_key}: {resource_id}")
                TORONTO_DATASETS[dataset_key]["resource_id"] = resource_id
                return resource_id
            
            return None
                
    except httpx.TimeoutException:
        logger.error(f"Timeout discovering resource ID for {dataset_key}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} error discovering resource ID for {dataset_key}: {e}")
        return None
    except httpx.HTTPError as e:
        logger.error(f"HTTP error discovering resource ID for {dataset_key}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error discovering resource ID for {dataset_key}: {e}")
        return None


def download_csv(url: str, output_path: Path, timeout: int = 300, is_datastore: bool = False, resource_id: Optional[str] = None) -> bool:
    """
    Download CSV file from URL or CKAN datastore.
    
    Args:
        url: URL to download from (or base URL if is_datastore=True)
        output_path: Path to save the file
        timeout: Request timeout in seconds
        is_datastore: If True, use /datastore/dump/ endpoint
        resource_id: Resource ID for datastore downloads
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Handle datastore_active resources
        if is_datastore and resource_id:
            # Use CKAN datastore dump endpoint
            base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
            datastore_url = f"{base_url}/datastore/dump/{resource_id}"
            logger.info(f"Downloading from datastore: {datastore_url}")
            download_url = datastore_url
        else:
            download_url = url
            logger.info(f"Downloading CSV from: {download_url}")
        
        logger.info(f"Saving to: {output_path}")
        
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(download_url)
            response.raise_for_status()
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Downloaded {file_size:.2f} MB successfully")
            return True
            
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading CSV from {download_url}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading CSV: Client error '{e.response.status_code} {e.response.reason_phrase}' for url '{download_url}'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{e.response.status_code}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"HTTP error downloading CSV: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading CSV: {e}")
        return False


def download_single_dataset(
    dataset_key: str,
    data_dir: Path,
    timestamp: str,
    skip_existing: bool = True,
    discover_if_missing: bool = True,
) -> Tuple[Optional[str], Optional[Path]]:
    """
    Download a single dataset.
    
    Args:
        dataset_key: Dataset key from TORONTO_DATASETS
        data_dir: Directory to save files
        timestamp: Timestamp string for filename
        skip_existing: Skip if file already exists
        discover_if_missing: Discover resource ID if not configured
        
    Returns:
        Tuple of (dataset_key, file_path) or (None, None) if failed
    """
    if dataset_key not in TORONTO_DATASETS:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return None, None
    
    dataset_info = TORONTO_DATASETS[dataset_key]
    logger.info(f"Processing dataset: {dataset_info['name']} ({dataset_key})")
    
    # Get resource ID
    resource_id = dataset_info.get("resource_id")
    direct_url = dataset_info.get("direct_url")
    is_datastore = dataset_info.get("datastore_active", False)
    
    # Discover resource ID if not configured
    if not resource_id and not direct_url and discover_if_missing:
        logger.info(f"Resource ID not configured for {dataset_key}, discovering...")
        discovered = discover_resource_id(dataset_key)
        if discovered:
            # Check if it's a URL or resource ID
            if discovered.startswith("http"):
                direct_url = discovered
            else:
                resource_id = discovered
                # Check if datastore_active was set during discovery
                is_datastore = dataset_info.get("datastore_active", False)
    
    # Construct URL
    if direct_url:
        url = direct_url
    elif resource_id:
        url = get_dataset_url(dataset_key, resource_id)
    else:
        logger.error(
            f"Could not determine download URL for {dataset_key}. "
            f"Resource ID not found and discovery failed."
        )
        return None, None
    
    # Generate filename
    filename = f"{dataset_key}_{timestamp}.csv"
    file_path = data_dir / filename
    
    # Skip if exists
    if skip_existing and file_path.exists():
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(
            f"File already exists: {file_path.name} ({file_size:.2f} MB). Skipping download."
        )
        return dataset_key, file_path
    
    # Download
    if download_csv(url, file_path, is_datastore=is_datastore, resource_id=resource_id):
        return dataset_key, file_path
    else:
        logger.error(f"Failed to download {dataset_key}")
        return None, None


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding using chardet or fallback method.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected encoding string
    """
    if HAS_CHARDET:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                # If confidence is low, fallback to utf-8
                if confidence < 0.7:
                    logger.debug(f"Low encoding confidence ({confidence:.2f}), using utf-8")
                    return 'utf-8'
                
                logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    else:
        # Fallback: try utf-8 first
        return 'utf-8'


def load_csv(file_path: Path, chunk_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load CSV file into pandas DataFrame with optimized encoding detection and chunked reading.
    
    Args:
        file_path: Path to CSV file
        chunk_size: If provided, process in chunks (for large files > 50MB)
        
    Returns:
        DataFrame or None if error
    """
    try:
        logger.info(f"Loading CSV: {file_path}")
        
        # Check file size for chunked reading
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        use_chunking = chunk_size is not None or file_size_mb > 50
        
        # Detect file format by checking magic bytes (Excel files start with PK)
        is_excel = False
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                # Excel files (XLSX) are ZIP archives, start with PK\x03\x04
                if header.startswith(b'PK\x03\x04') or file_path.suffix.lower() in ['.xlsx', '.xls']:
                    is_excel = True
        except Exception:
            pass
        
        # Handle Excel files
        if is_excel:
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info(f"Successfully loaded Excel file with openpyxl")
                logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                return df
            except ImportError:
                logger.warning("openpyxl not installed, trying xlrd")
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                    logger.info(f"Successfully loaded Excel file with xlrd")
                    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                    return df
                except Exception as excel_error:
                    logger.error(f"Failed to read Excel file: {excel_error}")
                    return None
            except Exception as excel_error:
                logger.error(f"Error reading Excel file: {excel_error}")
                return None
        
        # Detect encoding once (optimized)
        encoding = detect_encoding(file_path)
        
        # Load CSV with detected encoding
        if use_chunking:
            chunk_size = chunk_size or 50000
            logger.info(f"Large file ({file_size_mb:.1f} MB), using chunked reading (chunk_size={chunk_size})...")
            chunks = []
            try:
                for chunk in pd.read_csv(
                    file_path,
                    encoding=encoding,
                    chunksize=chunk_size,
                    low_memory=False,
                    parse_dates=False,
                    on_bad_lines='skip',
                    engine='c',
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded CSV with encoding: {encoding} (chunked)")
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.warning(f"Error with detected encoding {encoding}, retrying with Python engine...")
                # Fallback to Python engine with error handling
                chunks = []
                for chunk in pd.read_csv(
                    file_path,
                    encoding=encoding,
                    chunksize=chunk_size,
                    parse_dates=False,
                    on_bad_lines='skip',
                    engine='python',
                    sep=',',
                    quotechar='"',
                    skipinitialspace=True,
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded CSV with encoding: {encoding} (chunked, Python engine)")
        else:
            # Try C engine first (faster)
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    low_memory=False,
                    parse_dates=False,
                    on_bad_lines='skip',
                    engine='c',
                )
                logger.info(f"Successfully loaded CSV with encoding: {encoding}")
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.warning(f"CSV parsing error with encoding {encoding}: {e}")
                logger.info("Retrying with Python engine and error handling...")
                # Retry with Python engine (more lenient)
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        parse_dates=False,
                        on_bad_lines='skip',
                        engine='python',
                        sep=',',
                        quotechar='"',
                        skipinitialspace=True,
                    )
                    logger.info(f"Successfully loaded CSV with encoding: {encoding} (Python engine)")
                except Exception as retry_error:
                    logger.error(f"Failed to load CSV: {retry_error}")
                    return None
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns[:10])}...")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None


def validate_csv_structure(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that CSV has required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        (is_valid, missing_columns)
    """
    is_valid, report = validate_csv(df)
    
    if not is_valid:
        logger.warning(f"CSV validation failed: {report['errors']}")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        # Extract missing columns from errors
        missing = []
        for error in report['errors']:
            if 'Missing required columns' in error:
                # Parse missing columns from error message
                missing_str = error.split(': ')[-1] if ': ' in error else ''
                missing = [col.strip() for col in missing_str.split(',')]
        return False, missing
    
    if report.get('warnings'):
        for warning in report['warnings']:
            logger.warning(warning)
    
    logger.info("CSV structure validation passed")
    return True, []


def get_database_connection():
    """Get database connection from environment variables."""
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
            raise ValueError("DATABASE_URL not set. Please set DATABASE_URL in your .env file.")
    
    return create_engine(database_url)


def check_file_ingested(file_name: str, engine) -> bool:
    """
    Check if file has already been ingested.
    
    Args:
        file_name: Name of the file
        engine: SQLAlchemy engine
        
    Returns:
        True if already ingested, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM ingestion_metadata WHERE file_name = :file_name"),
                {"file_name": file_name}
            )
            count = result.fetchone()[0]
            return count > 0
    except Exception as e:
        logger.warning(f"Error checking ingestion metadata: {e}")
        return False


def discover_all_resources_parallel(datasets: list[str], max_workers: int = 5) -> Dict[str, Optional[str]]:
    """
    Discover all resource IDs in parallel before downloads.
    
    Args:
        datasets: List of dataset keys to discover
        max_workers: Maximum number of parallel discovery requests
        
    Returns:
        Dict mapping dataset_key -> resource_id (or None if failed)
    """
    logger.info(f"Discovering resource IDs for {len(datasets)} datasets in parallel (max {max_workers} workers)...")
    discovered = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(discover_resource_id, dataset_key): dataset_key
            for dataset_key in datasets
        }
        
        for future in as_completed(futures):
            dataset_key = futures[future]
            try:
                resource_id = future.result()
                discovered[dataset_key] = resource_id
                if resource_id:
                    logger.info(f"✓ Discovered resource for {dataset_key}")
                else:
                    logger.warning(f"✗ Failed to discover resource for {dataset_key}")
            except Exception as e:
                logger.error(f"✗ Exception discovering {dataset_key}: {e}")
                discovered[dataset_key] = None
    
    success_count = sum(1 for v in discovered.values() if v is not None)
    logger.info(f"Resource discovery complete: {success_count}/{len(datasets)} succeeded")
    return discovered


def batch_record_ingestion_metadata(
    files: Dict[str, Path],
    engine,
    dataset_types: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """
    Batch record ingestion metadata for multiple files in a single transaction.
    
    Args:
        files: Dict mapping dataset_key -> file_path
        engine: SQLAlchemy engine
        dataset_types: Optional dict mapping dataset_key -> dataset_type
        
    Returns:
        Dict mapping dataset_key -> success (bool)
    """
    if not engine:
        return {key: True for key in files.keys()}
    
    results = {}
    try:
        with engine.begin() as conn:
            for dataset_key, file_path in files.items():
                file_name = file_path.name
                dataset_type = (dataset_types or {}).get(dataset_key)
                
                try:
                    # Check if already ingested and insert/update in one query
                    result = conn.execute(
                        text("""
                            INSERT INTO ingestion_metadata 
                            (file_name, record_count, status, ingested_at, dataset_type)
                            VALUES (:file_name, 0, 'in_progress', CURRENT_TIMESTAMP, :dataset_type)
                            ON CONFLICT (file_name) 
                            DO UPDATE SET 
                                status = 'in_progress',
                                ingested_at = CURRENT_TIMESTAMP,
                                dataset_type = COALESCE(:dataset_type, ingestion_metadata.dataset_type)
                            RETURNING id
                        """),
                        {
                            "file_name": file_name,
                            "dataset_type": dataset_type,
                        }
                    )
                    metadata_id = result.fetchone()[0]
                    results[dataset_key] = True
                    logger.debug(f"Recorded ingestion start for {file_name} (id: {metadata_id})")
                except Exception as e:
                    logger.error(f"Error recording ingestion start for {file_name}: {e}")
                    results[dataset_key] = False
        
        logger.info(f"Batch recorded ingestion metadata for {sum(results.values())}/{len(files)} files")
        return results
    except Exception as e:
        logger.error(f"Error in batch recording: {e}")
        return {key: False for key in files.keys()}


def batch_update_ingestion_complete(
    files: Dict[str, Tuple[Path, int]],
    engine,
    dataset_types: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """
    Batch update ingestion completion for multiple files.
    
    Args:
        files: Dict mapping dataset_key -> (file_path, record_count)
        engine: SQLAlchemy engine
        dataset_types: Optional dict mapping dataset_key -> dataset_type
        
    Returns:
        Dict mapping dataset_key -> success (bool)
    """
    if not engine:
        return {key: True for key in files.keys()}
    
    results = {}
    try:
        with engine.begin() as conn:
            for dataset_key, (file_path, record_count) in files.items():
                file_name = file_path.name
                dataset_type = (dataset_types or {}).get(dataset_key)
                
                try:
                    conn.execute(
                        text("""
                            UPDATE ingestion_metadata
                            SET 
                                record_count = :record_count,
                                status = 'completed',
                                error_message = NULL,
                                dataset_type = COALESCE(:dataset_type, dataset_type)
                            WHERE file_name = :file_name
                        """),
                        {
                            "file_name": file_name,
                            "record_count": record_count,
                            "dataset_type": dataset_type,
                        }
                    )
                    results[dataset_key] = True
                except Exception as e:
                    logger.error(f"Error updating completion for {file_name}: {e}")
                    results[dataset_key] = False
        
        logger.info(f"Batch updated completion for {sum(results.values())}/{len(files)} files")
        return results
    except Exception as e:
        logger.error(f"Error in batch update: {e}")
        return {key: False for key in files.keys()}


def process_all_files_parallel(
    downloaded: Dict[str, Path],
    engine: Optional[Any],
    dataset_types: Optional[Dict[str, str]] = None,
    max_workers: int = 4,
) -> Dict[str, bool]:
    """
    Process downloaded files in parallel.
    
    Args:
        downloaded: Dict mapping dataset_key -> file_path
        engine: Database engine (optional)
        dataset_types: Optional dict mapping dataset_key -> dataset_type
        max_workers: Maximum number of parallel processing workers
        
    Returns:
        Dict mapping dataset_key -> success (bool)
    """
    logger.info(f"Processing {len(downloaded)} files in parallel (max {max_workers} workers)...")
    
    # Batch record ingestion start
    if engine:
        batch_record_ingestion_metadata(downloaded, engine, dataset_types)
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file_path,
                dataset_type=(dataset_types or {}).get(dataset_key),
                engine=None,  # Don't pass engine to avoid duplicate DB calls
            ): dataset_key
            for dataset_key, file_path in downloaded.items()
        }
        
        for future in as_completed(futures):
            dataset_key = futures[future]
            try:
                success = future.result()
                results[dataset_key] = success
                if success:
                    logger.success(f"✓ Successfully processed {dataset_key}")
                else:
                    logger.error(f"✗ Failed to process {dataset_key}")
            except Exception as e:
                logger.error(f"✗ Exception processing {dataset_key}: {e}")
                results[dataset_key] = False
    
    # Batch update completion with record counts
    if engine:
        file_records = {}
        for dataset_key, file_path in downloaded.items():
            if results.get(dataset_key, False):
                try:
                    df = load_csv(file_path)
                    record_count = len(df) if df is not None else 0
                    file_records[dataset_key] = (file_path, record_count)
                except Exception as e:
                    logger.warning(f"Could not get record count for {dataset_key}: {e}")
                    file_records[dataset_key] = (file_path, 0)
        
        if file_records:
            batch_update_ingestion_complete(file_records, engine, dataset_types)
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Processing complete: {success_count}/{len(downloaded)} files succeeded")
    return results


def download_all_datasets_parallel(
    datasets: list[str],
    data_dir: Path,
    max_workers: int = 5,
    skip_existing: bool = True,
) -> Dict[str, Path]:
    """
    Download multiple datasets in parallel to avoid bottlenecks.
    
    Args:
        datasets: List of dataset keys to download
        data_dir: Directory to save files
        max_workers: Maximum number of parallel downloads
        skip_existing: Skip if file already exists
        
    Returns:
        Dict mapping dataset_key -> file_path for successfully downloaded files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloaded = {}
    failed = []
    
    logger.info(f"Starting parallel download of {len(datasets)} datasets (max {max_workers} workers)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_dataset = {
            executor.submit(
                download_single_dataset,
                dataset_key,
                data_dir,
                timestamp,
                skip_existing,
                discover_if_missing=True,
            ): dataset_key
            for dataset_key in datasets
        }
        
        # Process completed downloads
        for future in as_completed(future_to_dataset):
            dataset_key = future_to_dataset[future]
            try:
                result_key, file_path = future.result()
                if result_key and file_path:
                    downloaded[result_key] = file_path
                    logger.success(f"✓ Successfully downloaded {result_key}")
                else:
                    failed.append(dataset_key)
                    logger.error(f"✗ Failed to download {dataset_key}")
            except Exception as e:
                failed.append(dataset_key)
                logger.error(f"✗ Exception downloading {dataset_key}: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Download Summary: {len(downloaded)} succeeded, {len(failed)} failed")
    logger.info("=" * 60)
    
    if failed:
        logger.warning(f"Failed datasets: {', '.join(failed)}")
    
    return downloaded


def record_ingestion_start(
    file_name: str, 
    engine, 
    record_count: int = 0,
    dataset_type: Optional[str] = None,
) -> Optional[int]:
    """
    Record ingestion start in metadata table.
    
    Args:
        file_name: Name of the file
        engine: SQLAlchemy engine
        record_count: Number of records (0 if not yet known)
        dataset_type: Dataset type/key (e.g., 'major-crime-indicators')
        
    Returns:
        metadata_id or None
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO ingestion_metadata 
                    (file_name, record_count, status, ingested_at, dataset_type)
                    VALUES (:file_name, :record_count, 'in_progress', CURRENT_TIMESTAMP, :dataset_type)
                    ON CONFLICT (file_name) 
                    DO UPDATE SET 
                        status = 'in_progress',
                        ingested_at = CURRENT_TIMESTAMP,
                        record_count = :record_count,
                        dataset_type = COALESCE(:dataset_type, ingestion_metadata.dataset_type)
                    RETURNING id
                """),
                {
                    "file_name": file_name, 
                    "record_count": record_count,
                    "dataset_type": dataset_type,
                }
            )
            metadata_id = result.fetchone()[0]
            conn.commit()
            return metadata_id
    except Exception as e:
        logger.error(f"Error recording ingestion start: {e}")
        return None


def record_ingestion_complete(
    file_name: str, 
    engine, 
    record_count: int, 
    last_timestamp: Optional[datetime] = None,
    dataset_type: Optional[str] = None,
) -> bool:
    """
    Record successful ingestion completion.
    
    Args:
        file_name: Name of the file
        engine: SQLAlchemy engine
        record_count: Number of records ingested
        last_timestamp: Last crime timestamp in the file
        dataset_type: Dataset type/key (e.g., 'major-crime-indicators')
        
    Returns:
        True if successful
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE ingestion_metadata
                    SET 
                        record_count = :record_count,
                        status = 'completed',
                        last_timestamp = :last_timestamp,
                        error_message = NULL,
                        dataset_type = COALESCE(:dataset_type, dataset_type)
                    WHERE file_name = :file_name
                """),
                {
                    "file_name": file_name,
                    "record_count": record_count,
                    "last_timestamp": last_timestamp,
                    "dataset_type": dataset_type,
                }
            )
            conn.commit()
            logger.info(f"Recorded ingestion completion: {record_count:,} records")
            return True
    except Exception as e:
        logger.error(f"Error recording ingestion completion: {e}")
        return False


def record_ingestion_error(file_name: str, engine, error_message: str) -> bool:
    """
    Record ingestion error.
    
    Args:
        file_name: Name of the file
        engine: SQLAlchemy engine
        error_message: Error message
        
    Returns:
        True if successful
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE ingestion_metadata
                    SET 
                        status = 'failed',
                        error_message = :error_message
                    WHERE file_name = :file_name
                """),
                {"file_name": file_name, "error_message": error_message[:1000]}
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error recording ingestion error: {e}")
        return False


def process_single_file(
    csv_path: Path,
    dataset_type: Optional[str] = None,
    engine: Optional[Any] = None,
) -> bool:
    """
    Process a single CSV file through Bronze layer.
    
    Args:
        csv_path: Path to CSV file
        dataset_type: Dataset type/key (e.g., 'major-crime-indicators')
        engine: Database engine (optional)
        
    Returns:
        True if successful, False otherwise
    """
    file_name = csv_path.name
    logger.info(f"Processing file: {file_name}")
    
    # Check if already ingested
    if engine:
        try:
            if check_file_ingested(file_name, engine):
                logger.warning(f"File {file_name} has already been ingested. Skipping.")
                logger.info("To re-ingest, delete the record from ingestion_metadata table.")
                return True
            
            # Record ingestion start
            record_ingestion_start(file_name, engine, record_count=0, dataset_type=dataset_type)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            logger.warning("Continuing without database tracking...")
            engine = None
    
    # Load CSV
    df = load_csv(csv_path)
    if df is None:
        logger.error("Failed to load CSV")
        if engine:
            record_ingestion_error(file_name, engine, "Failed to load CSV file")
        return False
    
    # Validate structure (warn but don't fail for different schemas)
    is_valid, missing = validate_csv_structure(df)
    if not is_valid:
        logger.warning(f"CSV validation warnings. Missing columns: {missing}")
        logger.info("Continuing with available columns...")
    
    # Update record count
    record_count = len(df)
    if engine:
        record_ingestion_start(file_name, engine, record_count=record_count, dataset_type=dataset_type)
    
    logger.info(f"Successfully loaded {record_count:,} records")
    logger.info(f"CSV saved to: {csv_path}")
    
    # Record completion (actual DB insertion happens in Gold layer)
    if engine:
        # Try to find last timestamp if possible
        last_timestamp = None
        date_cols = [col for col in df.columns if "date" in col.lower() or "occurrence" in col.lower()]
        if date_cols:
            try:
                # This is a placeholder - actual parsing happens in Silver/Gold layers
                pass
            except:
                pass
        
        record_ingestion_complete(file_name, engine, record_count, last_timestamp, dataset_type=dataset_type)
    
    return True


def main(
    csv_url: Optional[str] = None,
    csv_file: Optional[str] = None,
    skip_download: bool = False,
    dataset: Optional[str] = None,
    all_datasets: bool = False,
    max_workers: int = 5,
):
    """
    Main ingestion function.
    
    Args:
        csv_url: URL to download CSV from (defaults to Toronto Open Data)
        csv_file: Path to existing CSV file (if not downloading)
        skip_download: If True, skip download and use existing file
        dataset: Specific dataset key to download (from TORONTO_DATASETS)
        all_datasets: If True, download all configured datasets in parallel
        max_workers: Maximum number of parallel downloads (when all_datasets=True)
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Bronze Layer: CSV Ingestion")
    logger.info("=" * 60)
    
    # Get data directory
    data_dir = get_data_directory()
    
    # Get database connection (optional)
    try:
        engine = get_database_connection()
    except Exception as e:
        logger.warning(f"Database connection error: {e}")
        logger.warning("Continuing without database tracking...")
        engine = None
    
    # Handle multi-dataset download
    if all_datasets:
        logger.info("Downloading all configured datasets in parallel...")
        all_dataset_keys = get_all_dataset_keys()
        
        # Discover all resource IDs in parallel first (optimization)
        logger.info("Step 1: Discovering resource IDs...")
        discovered = discover_all_resources_parallel(all_dataset_keys, max_workers=max_workers)
        
        # Update TORONTO_DATASETS with discovered resource IDs
        for dataset_key, resource_id in discovered.items():
            if resource_id:
                if resource_id.startswith("http"):
                    TORONTO_DATASETS[dataset_key]["direct_url"] = resource_id
                else:
                    TORONTO_DATASETS[dataset_key]["resource_id"] = resource_id
        
        # Download all datasets in parallel
        logger.info("Step 2: Downloading datasets...")
        downloaded = download_all_datasets_parallel(
            all_dataset_keys,
            data_dir,
            max_workers=max_workers,
            skip_existing=skip_download,
        )
        
        # Process all downloaded files in parallel (optimization)
        logger.info("Step 3: Processing downloaded files...")
        dataset_types = {key: key for key in downloaded.keys()}
        results = process_all_files_parallel(
            downloaded,
            engine,
            dataset_types=dataset_types,
            max_workers=max_workers,
        )
        
        success_count = sum(1 for v in results.values() if v)
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Bronze layer complete! Processed {success_count}/{len(downloaded)} files")
        logger.info("=" * 60)
        logger.info(f"Next step: Run Silver layer to clean the data")
        return
    
    # Handle single dataset
    dataset_type = None
    if dataset and dataset in TORONTO_DATASETS:
        dataset_type = dataset
        # Try to get URL for this dataset
        dataset_info = get_dataset_info(dataset)
        if dataset_info:
            resource_id = dataset_info.get("resource_id")
            if not resource_id:
                # Discover resource ID
                logger.info(f"Discovering resource ID for {dataset}...")
                resource_id = discover_resource_id(dataset)
            
            if resource_id:
                csv_url = get_dataset_url(dataset, resource_id)
            elif dataset_info.get("direct_url"):
                csv_url = dataset_info["direct_url"]
    
    # Determine file path
    if csv_file:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
    else:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_type:
            csv_path = data_dir / f"{dataset_type}_{timestamp}.csv"
        else:
            csv_path = data_dir / f"toronto_crime_{timestamp}.csv"
        
        # Download if needed
        if not skip_download:
            url = csv_url or TORONTO_CRIME_DATA_URL
            if not download_csv(url, csv_path):
                logger.error("Failed to download CSV")
                sys.exit(1)
        elif not csv_path.exists():
            logger.error(f"CSV file not found and download skipped: {csv_path}")
            sys.exit(1)
    
    # Process single file
    if process_single_file(csv_path, dataset_type=dataset_type, engine=engine):
        logger.info("=" * 60)
        logger.info("Bronze layer ingestion complete!")
        logger.info("=" * 60)
        logger.info(f"Next step: Run Silver layer to clean the data")
        logger.info(f"CSV file ready at: {csv_path}")
    else:
        logger.error("Failed to process file")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bronze layer CSV ingestion")
    parser.add_argument(
        "--url",
        type=str,
        help="URL to download CSV from (defaults to Toronto Open Data)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to existing CSV file (skip download)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and use existing file in data/raw/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset key to download (e.g., 'major-crime-indicators')",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Download all configured datasets in parallel",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel downloads (default: 5)",
    )
    
    args = parser.parse_args()
    
    main(
        csv_url=args.url,
        csv_file=args.file,
        skip_download=args.skip_download,
        dataset=args.dataset,
        all_datasets=args.all_datasets,
        max_workers=args.max_workers,
    )

