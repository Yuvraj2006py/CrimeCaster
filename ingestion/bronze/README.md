# Bronze Layer: CSV Ingestion

The Bronze layer downloads and loads raw CSV files from the Toronto Open Data Portal. 
Supports multiple datasets with automatic CKAN API discovery and parallel downloads.

## Supported Datasets

The Bronze layer supports 9 Toronto Open Data datasets:

1. **major-crime-indicators** - Major Crime Indicators (pre-configured)
2. **police-annual-statistical-report-shooting-occurrences** - Shooting Occurrences
3. **wellbeing-toronto-safety** - Wellbeing Toronto Safety
4. **police-annual-statistical-report-reported-crimes** - Reported Crimes
5. **police-annual-statistical-report-miscellaneous-data** - Miscellaneous Police Data
6. **police-annual-statistical-report-homicide** - Homicide Data
7. **police-annual-statistical-report-miscellaneous-firearms** - Miscellaneous Firearms
8. **theft-from-motor-vehicle** - Theft from Motor Vehicle
9. **shootings-firearm-discharges** - Shootings and Firearm Discharges

Resource IDs are automatically discovered via CKAN API if not pre-configured.

## Usage

### Command Line

```bash
# Download ALL datasets in parallel (recommended)
python -m ingestion.bronze.csv_loader --all-datasets

# Download specific dataset
python -m ingestion.bronze.csv_loader --dataset major-crime-indicators
python -m ingestion.bronze.csv_loader --dataset shootings-firearm-discharges

# Download with custom parallelism (default: 5 workers)
python -m ingestion.bronze.csv_loader --all-datasets --max-workers 10

# Legacy: Download from default URL (major-crime-indicators)
python -m ingestion.bronze.csv_loader

# Use a custom URL
python -m ingestion.bronze.csv_loader --url "https://example.com/data.csv"

# Use an existing CSV file
python -m ingestion.bronze.csv_loader --file "data/raw/my_file.csv"

# Skip download (use most recent file in data/raw/)
python -m ingestion.bronze.csv_loader --skip-download
```

### Using Docker

```bash
# Run ingestion container
docker-compose run --rm ingestion

# Or with custom file
docker-compose run --rm ingestion --file "data/raw/my_file.csv"
```

### Python API

```python
from ingestion.bronze.csv_loader import main

# Download and process
main()

# Use existing file
main(csv_file="data/raw/my_file.csv", skip_download=True)
```

## What It Does

1. **Discovers Resource IDs** - Automatically discovers CSV resource IDs via CKAN API
2. **Downloads CSV** - Downloads from Toronto Open Data Portal (or custom URL)
   - Supports parallel downloads to avoid bottlenecks (default: 5 concurrent)
   - Automatically handles rate limiting and retries
3. **Saves to** `data/raw/` directory with timestamp and dataset key
4. **Validates** CSV structure (required columns, data types)
5. **Tracks** ingestion in `ingestion_metadata` table with `dataset_type` field
6. **Prevents** duplicate ingestion of same file

## Performance

- **Parallel Downloads**: Downloads multiple datasets simultaneously (configurable with `--max-workers`)
- **No Bottlenecks**: Uses ThreadPoolExecutor for concurrent downloads
- **Automatic Discovery**: Resource IDs discovered on-the-fly via CKAN API
- **Error Recovery**: Failed downloads don't block other datasets

## Expected CSV Format

The CSV should have these columns (name variations are supported):

- **Required:**
  - `occurrence_date` or `Occurrence Date` - Date of crime
  - `lat` or `latitude` - Latitude coordinate
  - `long` or `longitude` - Longitude coordinate
  - `major_crime_indicator` or `MCI_CATEGORY` - Type of crime

- **Optional:**
  - `neighbourhood` - Neighbourhood name
  - `premises_type` - Type of location
  - `occurrence_time` - Time of occurrence

## Output

- CSV file saved to `data/raw/toronto_crime_YYYYMMDD_HHMMSS.csv`
- Metadata recorded in `ingestion_metadata` table
- Logs show validation results and record counts

## Next Steps

After Bronze layer completes, run:
1. **Silver layer** - Clean and normalize the data
2. **Gold layer** - Map to H3 hexagons and load into database

