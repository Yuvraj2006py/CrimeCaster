# Silver Layer: Data Cleaning

The Silver layer cleans and normalizes raw data from the Bronze layer, preparing it for the Gold layer transformation.

## What It Does

1. **Removes duplicates** - Based on EVENT_UNIQUE_ID
2. **Validates coordinates** - Filters out invalid Toronto coordinates
3. **Cleans data** - Standardizes formats and handles missing values
4. **Transforms schema** - Maps to target database schema
5. **Saves cleaned data** - Outputs to `data/silver/` directory

## Usage

### Command Line

```bash
# Use latest Bronze file automatically
python -m ingestion.silver.cleaner

# Specify input file
python -m ingestion.silver.cleaner --input "data/raw/my_file.csv"

# Specify output file
python -m ingestion.silver.cleaner --output "data/silver/my_cleaned_file.csv"

# Skip coordinate validation (not recommended)
python -m ingestion.silver.cleaner --skip-validation
```

### Using Docker

```bash
# Run Silver layer cleaning
docker-compose run --rm ingestion python -m ingestion.silver.cleaner
```

### Python API

```python
from ingestion.silver.cleaner import main

# Clean latest Bronze file
main()

# Clean specific file
main(input_file="data/raw/my_file.csv")
```

## Cleaning Steps

### 1. Duplicate Removal
- Removes duplicate records based on `EVENT_UNIQUE_ID`
- Keeps first occurrence
- Typically removes ~12-15% of records

### 2. Coordinate Validation
- Validates latitude: 43.5 to 43.9 (Toronto bounds)
- Validates longitude: -79.8 to -79.0 (Toronto bounds)
- Removes rows with invalid or missing coordinates
- Typically removes ~1-2% of records

### 3. Data Transformation
- **Date parsing**: Combines `OCC_DATE` + `OCC_HOUR` → `occurred_at` (datetime)
- **Coordinates**: Maps `LAT_WGS84` → `latitude`, `LONG_WGS84` → `longitude`
- **Crime types**: Standardizes `MCI_CATEGORY` → `crime_type`
- **Neighbourhood**: Cleans `NEIGHBOURHOOD_158` → `neighbourhood`
- **Premise type**: Maps `PREMISES_TYPE` → `premise_type`
- **Source file**: Records original CSV filename

### 4. Quality Checks
- Removes rows with missing critical fields (date, coordinates, crime type)
- Ensures all required fields are present
- Validates data types

## Output Schema

The cleaned data has these columns:

| Column | Type | Description |
|--------|------|-------------|
| `occurred_at` | datetime | Crime occurrence timestamp |
| `latitude` | float64 | Latitude coordinate (WGS84) |
| `longitude` | float64 | Longitude coordinate (WGS84) |
| `crime_type` | string | Standardized crime type |
| `neighbourhood` | string | Neighbourhood name (nullable) |
| `premise_type` | string | Premise type (nullable) |
| `source_file` | string | Original CSV filename |

## Data Quality

After cleaning, you can expect:
- **No nulls** in critical fields (occurred_at, latitude, longitude, crime_type)
- **Valid coordinates** within Toronto bounds
- **No duplicates** (based on unique ID)
- **Standardized formats** for all fields

## Typical Results

For a typical Toronto crime dataset:
- **Input**: ~450,000 rows
- **Duplicates removed**: ~12-15% (58,000-65,000 rows)
- **Invalid coordinates**: ~1-2% (5,000-6,000 rows)
- **Output**: ~385,000-390,000 clean rows

## Next Steps

After Silver layer completes, run:
1. **Gold layer** - Map to H3 hexagons and load into database

## Troubleshooting

### "Coordinate columns not found"
- Check that your CSV has `LAT_WGS84` and `LONG_WGS84` columns
- Or columns named `lat`/`latitude` and `long`/`longitude`

### "Date column not found"
- Ensure CSV has `OCC_DATE` or similar date column
- Check column names match expected patterns

### Too many rows removed
- Check coordinate bounds are correct for your data
- Review logs to see why rows were filtered
- Use `--skip-validation` to bypass coordinate checks (not recommended)

