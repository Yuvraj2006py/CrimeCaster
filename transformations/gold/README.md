# Gold Layer: H3 Mapping and Database Loading

The Gold layer is the final transformation step that maps cleaned data to H3 hexagons and loads it into PostgreSQL. This makes the data available for feature engineering and ML training.

## What It Does

1. **Loads cleaned data** from Silver layer CSV
2. **Maps coordinates to H3 hexagons** (resolution 9 = ~300m hexagons)
3. **Validates and prepares data** for database insertion
4. **Batch inserts into PostgreSQL** (`crimes` table)
5. **Updates ingestion metadata** to track completion
6. **Verifies insertion** to ensure data integrity

## Usage

### Command Line

```bash
# Use latest Silver file automatically
python -m transformations.gold.h3_mapper

# Specify input file
python -m transformations.gold.h3_mapper --input "data/silver/cleaned_crime_data_*.csv"

# Custom batch size (for performance tuning)
python -m transformations.gold.h3_mapper --batch-size 5000

# Force reload even if records exist
python -m transformations.gold.h3_mapper --no-skip-existing

# Skip verification (faster, but less safe)
python -m transformations.gold.h3_mapper --no-verify
```

### Using Docker

```bash
# Run Gold layer
docker-compose run --rm ingestion python -m transformations.gold.h3_mapper
```

### Python API

```python
from transformations.gold.h3_mapper import main

# Load latest Silver file
main()

# Load specific file
main(input_file="data/silver/cleaned_crime_data_*.csv")
```

## H3 Hexagon Mapping

### Resolution 9
- **Resolution**: 9
- **Average hexagon area**: ~0.09 km² (~300m edge length)
- **Purpose**: City-level crime analysis
- **Coverage**: Entire Toronto area

### How It Works

For each crime record:
1. Extract latitude and longitude
2. Convert to H3 hexagon index using `h3.latlng_to_cell()`
3. Store H3 index in `h3_index` column
4. PostGIS geometry is auto-created by database trigger

### Example

```python
import h3

# Toronto coordinates
lat, lon = 43.6532, -79.3832

# Get H3 hexagon
h3_index = h3.latlng_to_cell(lat, lon, resolution=9)
# Result: "892b9bc46d7ffff"
```

## Database Schema

The Gold layer inserts into the `crimes` table with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL | Auto-generated primary key |
| `crime_type` | VARCHAR(100) | Type of crime (from MCI_CATEGORY) |
| `occurred_at` | TIMESTAMP WITH TIME ZONE | When crime occurred |
| `latitude` | DECIMAL(10, 8) | Latitude coordinate |
| `longitude` | DECIMAL(11, 8) | Longitude coordinate |
| `neighbourhood` | VARCHAR(100) | Neighbourhood name (nullable) |
| `premise_type` | VARCHAR(100) | Premise type (nullable) |
| `h3_index` | VARCHAR(20) | H3 hexagon index (resolution 9) |
| `source_file` | VARCHAR(255) | Original CSV filename |
| `created_at` | TIMESTAMP WITH TIME ZONE | Auto-generated timestamp |
| `geom` | GEOMETRY(POINT, 4326) | PostGIS geometry (auto-created) |

## Performance

### Batch Insertion
- **Default batch size**: 10,000 rows
- **Optimization**: Uses pandas `to_sql()` with `method='multi'` for bulk inserts
- **Transaction management**: Each batch is a separate transaction
- **Error recovery**: Failed batches are retried with individual inserts

### Typical Performance
- **Small files** (< 100K rows): ~1-2 minutes
- **Medium files** (100K-500K rows): ~5-10 minutes
- **Large files** (> 500K rows): ~15-30 minutes

Performance depends on:
- Database connection speed
- Network latency
- Database load
- Batch size

## Error Handling

### Automatic Recovery
- **Failed batches**: Automatically retried with individual row inserts
- **Invalid coordinates**: Filtered out (shouldn't happen after Silver layer)
- **Database errors**: Logged with detailed error messages
- **Transaction rollback**: Failed batches don't corrupt database

### Common Issues

**"Failed to connect to database"**
- Check `DATABASE_URL` in `.env` file
- Verify database is accessible
- Test connection: `python scripts/test_connection.py`

**"Duplicate key error"**
- Records already exist for this source file
- Use `--no-skip-existing` to force reload
- Or delete existing records first

**"H3 mapping failed"**
- Invalid coordinates (should be filtered in Silver layer)
- Check coordinate validation in Silver layer

## Verification

After insertion, the Gold layer automatically verifies:
- ✅ Record count matches expected
- ✅ All records have H3 indices
- ✅ All records have PostGIS geometry
- ✅ Ingestion metadata is updated

## Incremental Loading

The Gold layer supports incremental loading:
- Checks `ingestion_metadata` table for existing records
- Skips files that are already loaded (unless `--no-skip-existing`)
- Updates metadata with new record counts

## Data Quality

### Pre-Insertion Checks
- ✅ No nulls in required fields
- ✅ Valid coordinates (Toronto bounds)
- ✅ Valid H3 indices
- ✅ Proper data types

### Post-Insertion Verification
- ✅ Record counts match
- ✅ H3 indices present
- ✅ PostGIS geometry created
- ✅ Metadata updated

## Next Steps

After Gold layer completes:
1. **Feature Engineering** - Generate ML features from `crimes` table
2. **Model Training** - Train ML models on engineered features
3. **API Endpoints** - Serve predictions via FastAPI

## Troubleshooting

### "No CSV files found in data/silver/"
- Run Silver layer first: `python -m ingestion.silver.cleaner`

### "Database connection failed"
- Verify `.env` file has correct `DATABASE_URL`
- Test connection: `python scripts/test_connection.py`
- Check Supabase project is active

### "Batch insertion failed"
- Check database logs for detailed error
- Try smaller batch size: `--batch-size 1000`
- Verify database has enough space

### "Verification failed"
- Check database for partial inserts
- Review logs for error messages
- May need to clean up and retry

## Best Practices

1. **Always verify** database insertion after loading
2. **Monitor progress** for large files (logs show progress)
3. **Use appropriate batch size** (10K default, adjust for your DB)
4. **Check ingestion metadata** to track what's been loaded
5. **Test with small sample** before loading full dataset

## Example Workflow

```bash
# 1. Run Bronze layer (download CSV)
python -m ingestion.bronze.csv_loader

# 2. Run Silver layer (clean data)
python -m ingestion.silver.cleaner

# 3. Run Gold layer (load to database)
python -m transformations.gold.h3_mapper

# 4. Verify data in database
python -c "
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM crimes'))
    print(f'Total crimes: {result.fetchone()[0]:,}')
"
```

