# Multi-Dataset Ingestion Implementation

## Overview

The Bronze layer has been enhanced to support downloading and ingesting **9 different Toronto Open Data datasets** with automatic CKAN API discovery and parallel downloads to prevent bottlenecks.

## Implemented Features

### ✅ CKAN API Discovery
- Automatically discovers resource IDs for datasets via CKAN API
- Falls back to direct URLs if resource IDs aren't available
- Handles errors gracefully with detailed logging
- Updates dataset configuration in-memory for subsequent downloads

### ✅ Parallel Downloads
- Downloads multiple datasets simultaneously using `ThreadPoolExecutor`
- Configurable parallelism (default: 5 workers, adjustable via `--max-workers`)
- No bottlenecks - downloads happen concurrently
- Failed downloads don't block other datasets

### ✅ Dataset Tracking
- Database schema updated with `dataset_type` column in `ingestion_metadata` table
- Each file is tagged with its dataset source
- Enables filtering and analysis by dataset type

### ✅ Backward Compatibility
- Original single-dataset functionality still works
- Legacy `TORONTO_CRIME_DATA_URL` still supported
- Existing scripts continue to work without changes

## Supported Datasets

1. **major-crime-indicators** (pre-configured)
2. **police-annual-statistical-report-shooting-occurrences**
3. **wellbeing-toronto-safety**
4. **police-annual-statistical-report-reported-crimes**
5. **police-annual-statistical-report-miscellaneous-data**
6. **police-annual-statistical-report-homicide**
7. **police-annual-statistical-report-miscellaneous-firearms**
8. **theft-from-motor-vehicle**
9. **shootings-firearm-discharges**

## Usage Examples

### Download All Datasets
```bash
python -m ingestion.bronze.csv_loader --all-datasets
```

### Download Specific Dataset
```bash
python -m ingestion.bronze.csv_loader --dataset shootings-firearm-discharges
```

### Parallel Downloads with Custom Workers
```bash
python -m ingestion.bronze.csv_loader --all-datasets --max-workers 10
```

### Test CKAN Discovery
```bash
python scripts/test_ckan_discovery.py
```

## Files Modified/Created

### New Files
- `ingestion/bronze/dataset_config.py` - Dataset configuration
- `scripts/test_ckan_discovery.py` - Test script for CKAN API discovery
- `docs/multi_dataset_ingestion.md` - This document

### Modified Files
- `ingestion/bronze/csv_loader.py` - Enhanced with multi-dataset support
- `ingestion/bronze/README.md` - Updated documentation
- `sql/schema.sql` - Added `dataset_type` column to `ingestion_metadata`

## Technical Details

### CKAN API Discovery
- Uses `package_show` API endpoint
- Prefers CSV format resources
- Falls back to any downloadable resource with a URL
- Handles both resource IDs and direct URLs

### Parallel Download Implementation
- Uses `ThreadPoolExecutor` with configurable max workers
- Each dataset download is independent
- Progress tracked per dataset
- Summary report at end

### Error Handling
- Timeout handling for API calls (30s default)
- HTTP error handling with detailed messages
- Graceful degradation if discovery fails
- Continues processing other datasets if one fails

## Database Schema Changes

```sql
ALTER TABLE ingestion_metadata 
ADD COLUMN dataset_type VARCHAR(100);

CREATE INDEX idx_ingestion_dataset_type ON ingestion_metadata(dataset_type);
```

The `dataset_type` column stores the dataset key (e.g., 'major-crime-indicators', 'shootings-firearm-discharges').

## Performance Considerations

- **No Bottlenecks**: Parallel downloads ensure maximum throughput
- **Configurable**: Adjust `--max-workers` based on network/server capacity
- **Efficient**: Resource IDs discovered once and cached in-memory
- **Resilient**: Failed downloads don't block others

## Testing

Run the test script to verify CKAN API discovery:
```bash
python scripts/test_ckan_discovery.py
```

This will:
- Test discovery for all 9 datasets
- Report which ones are already configured
- Report which ones were successfully discovered
- Report any failures

## Next Steps

After downloading all datasets:
1. Run Silver layer to clean each dataset
2. Run Gold layer to map to H3 and load into database
3. Each dataset will be tracked separately in `ingestion_metadata`

