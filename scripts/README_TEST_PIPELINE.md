# Test Pipeline

A lightweight test pipeline for validating the end-to-end data flow with a small dataset.

## Purpose

The test pipeline allows you to:
- Test the complete pipeline (Bronze → Silver → Gold) with a small dataset
- Identify errors quickly without processing large amounts of data
- Verify database connections and data transformations
- Debug issues in a controlled environment

## Usage

### Basic Usage

```bash
# Run test pipeline with default settings (1000 rows from major-crime-indicators)
python scripts/test_pipeline.py
```

### Custom Options

```bash
# Test with 500 rows
python scripts/test_pipeline.py --rows 500

# Test with a different dataset
python scripts/test_pipeline.py --dataset shootings-firearm-discharges

# Test with 2000 rows from a specific dataset
python scripts/test_pipeline.py --rows 2000 --dataset major-crime-indicators
```

### Skip Layers

```bash
# Skip Bronze layer (use existing test file)
python scripts/test_pipeline.py --skip-bronze

# Skip Silver layer
python scripts/test_pipeline.py --skip-silver

# Skip Gold layer
python scripts/test_pipeline.py --skip-gold
```

### Verification Only

```bash
# Only verify existing test data in database
python scripts/test_pipeline.py --verify-only
```

## What It Does

1. **Bronze Layer (Download)**
   - Downloads the specified dataset
   - Limits to the first N rows (default: 1000)
   - Saves as `test_{dataset}_{timestamp}_limited.csv`
   - Validates the data structure

2. **Silver Layer (Cleaning)**
   - Cleans and normalizes the test data
   - Validates coordinates
   - Standardizes formats
   - Saves as `cleaned_test_{dataset}_{timestamp}.csv`

3. **Gold Layer (H3 Mapping & Database)**
   - Maps coordinates to H3 hexagons
   - Loads data into PostgreSQL database
   - Uses smaller batch size (500 rows) for testing
   - Verifies insertion

4. **Verification**
   - Queries database for test records
   - Reports statistics (total records, unique hexagons, date range)
   - Confirms data was loaded correctly

## Output

The test pipeline provides:
- ✅ Clear progress indicators for each phase
- ❌ Detailed error messages if something fails
- 📊 Summary statistics at the end
- 🔍 Database verification results

## Example Output

```
================================================================================
TEST PIPELINE - STARTING
Started at: 2024-01-15 10:30:00
Dataset: major-crime-indicators
Max rows: 1,000
================================================================================

================================================================================
TEST PIPELINE: BRONZE LAYER - Downloading Test Data
================================================================================
Dataset: major-crime-indicators
Max rows: 1,000

Downloading full dataset...
Downloaded to: data/raw/test_major-crime-indicators_20240115_103000_full.csv
Loading CSV and limiting to 1,000 rows...
Loaded 1,000 rows, 15 columns
✅ Created test dataset: test_major-crime-indicators_20240115_103000_limited.csv (1,000 rows)

✅ Bronze layer validation complete

================================================================================
TEST PIPELINE: SILVER LAYER - Data Cleaning
================================================================================
Found 1 test file(s) to process
Processing: test_major-crime-indicators_20240115_103000_limited.csv
✅ Processed: test_major-crime-indicators_20240115_103000_limited.csv
✅ Silver layer complete

================================================================================
TEST PIPELINE: GOLD LAYER - H3 Mapping & Database Loading
================================================================================
Found 1 test Silver file(s) to process
Processing: cleaned_test_major-crime-indicators_20240115_103000_limited.csv
✅ Processed: cleaned_test_major-crime-indicators_20240115_103000_limited.csv
✅ Gold layer complete

================================================================================
TEST PIPELINE: VERIFICATION
================================================================================
Total test records: 1,000
Unique source files: 1
Unique H3 hexagons: 245
Earliest crime: 2023-01-01 00:00:00+00:00
Latest crime: 2023-12-31 23:59:59+00:00
✅ Test data found in database

================================================================================
TEST PIPELINE SUMMARY
================================================================================
Duration: 45.23 seconds (0.75 minutes)
Completed at: 2024-01-15 10:30:45

Bronze Download: ✅ SUCCESS
Bronze Process: ✅ SUCCESS
Silver: ✅ SUCCESS
Gold: ✅ SUCCESS
Verification: ✅ SUCCESS

🎉 Test pipeline completed successfully!
```

## Troubleshooting

### No test data in database
- Check that DATABASE_URL is set correctly in `.env`
- Verify database connection is working
- Check if data was actually inserted (look for errors in Gold layer)

### Bronze layer fails
- Check internet connection
- Verify dataset key is correct
- Check if Toronto Open Data Portal is accessible

### Silver layer fails
- Check if Bronze file has valid data
- Verify coordinates are present
- Check for encoding issues

### Gold layer fails
- Check database connection
- Verify PostGIS extension is enabled
- Check database storage limits (Neon free tier: 512 MB)
- Review error messages for specific issues

## Next Steps

After successful test pipeline:
1. Review the database to verify data quality
2. Check logs for any warnings
3. Run full pipeline: `python scripts/run_pipeline.py`
4. Clean up test files if desired (they're prefixed with `test_`)

## Files Created

Test files are prefixed with `test_`:
- Bronze: `data/raw/test_{dataset}_{timestamp}_limited.csv`
- Silver: `data/silver/cleaned_test_{dataset}_{timestamp}_limited.csv`
- Database records with `source_file LIKE 'test_%'`

You can safely delete these files after testing.

