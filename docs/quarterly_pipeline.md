# Quarterly Data Pipeline

## Overview

The data pipeline runs automatically every quarter using GitHub Actions (free for public repositories). The pipeline downloads, cleans, and loads Toronto crime data into the database.

## Schedule

- **Frequency**: Quarterly (4 times per year)
- **Dates**: January 1, April 1, July 1, October 1
- **Time**: 2:00 AM UTC (9:00 PM EST / 10:00 PM EDT previous day)
- **Platform**: GitHub Actions (free tier)

## Pipeline Phases

The pipeline consists of three phases that run sequentially:

### 1. Bronze Layer - CSV Download
- Downloads all 9 Toronto Open Data datasets in parallel
- Saves raw CSV files to `data/raw/`
- Tracks ingestion metadata in database

### 2. Silver Layer - Data Cleaning
- Removes duplicate records
- Validates coordinates (Toronto bounds)
- Standardizes data formats
- Saves cleaned CSV files to `data/silver/`

### 3. Gold Layer - H3 Mapping & Database Loading
- Maps coordinates to H3 hexagons (resolution 9, ~300m)
- Batch inserts into PostgreSQL database
- Creates PostGIS geometry automatically
- Verifies data integrity

## Manual Execution

### Local Testing

Run the complete pipeline locally:

```bash
# From project root
python scripts/run_pipeline.py
```

Or run individual layers:

```bash
# Bronze layer only
python -m ingestion.bronze.csv_loader --all-datasets

# Silver layer only
python -m ingestion.silver.cleaner

# Gold layer only
python -m transformations.gold.h3_mapper
```

### GitHub Actions Manual Trigger

1. Go to your GitHub repository
2. Click on **Actions** tab
3. Select **Quarterly Data Pipeline** workflow
4. Click **Run workflow** button
5. Select branch (usually `main`)
6. Click **Run workflow**

## Monitoring

### Check Pipeline Status

1. Go to **Actions** tab in GitHub
2. Find the latest workflow run
3. Click on the run to see detailed logs
4. Check each step for success/failure

### View Logs

- Logs are automatically uploaded as artifacts
- Download from the workflow run page
- Logs are retained for 30 days
- Local logs are saved to `logs/pipeline_YYYY-MM-DD.log`

### Pipeline Timeout

- Default timeout: 60 minutes
- Typical runtime: 20-40 minutes
- If timeout occurs, check logs for bottlenecks

## Configuration

### Required Secrets

The pipeline requires one GitHub secret:

- **DATABASE_URL**: PostgreSQL connection string from Neon
  - Format: `postgresql://user:password@host:port/database?sslmode=require`
  - Set in: Repository Settings → Secrets and variables → Actions

### Cron Schedule

The cron expression is: `0 2 1 1,4,7,10 *`

- `0` - Minute (0th minute)
- `2` - Hour (2 AM UTC)
- `1` - Day of month (1st)
- `1,4,7,10` - Months (Jan, Apr, Jul, Oct)
- `*` - Day of week (any)

To modify the schedule, edit `.github/workflows/quarterly-pipeline.yml`

## Troubleshooting

### Pipeline Fails at Bronze Layer

**Symptoms**: Downloads fail or timeout

**Solutions**:
- Check internet connectivity (GitHub Actions runner)
- Verify Toronto Open Data API is accessible
- Check for rate limiting (reduce `--max-workers`)
- Review logs for specific error messages

### Pipeline Fails at Silver Layer

**Symptoms**: Data cleaning errors

**Solutions**:
- Verify Bronze layer output exists in `data/raw/`
- Check file permissions
- Review data format changes from source
- Check coordinate validation bounds

### Pipeline Fails at Gold Layer

**Symptoms**: Database connection or insertion errors

**Solutions**:
- Verify `DATABASE_URL` secret is correct
- Check database is accessible from GitHub Actions
- Verify database has enough storage space
- Check for duplicate records (use `--no-skip-existing` to force)
- Review H3 mapping errors in logs

### Workflow Doesn't Run

**Symptoms**: No workflow runs on scheduled date

**Solutions**:
- Verify repository is **public** (required for free GitHub Actions)
- Check cron syntax is correct
- Verify workflow file is in `.github/workflows/`
- Check GitHub Actions is enabled: Settings → Actions → General
- Verify you haven't exceeded free tier limits (2,000 min/month)

### Database Connection Issues

**Symptoms**: Connection timeout or authentication errors

**Solutions**:
- Verify `DATABASE_URL` includes `?sslmode=require`
- Check Neon database is not paused
- Verify IP allowlist (if enabled) includes GitHub Actions IPs
- Test connection string locally first

## Cost

### Free Tier Usage

- **GitHub Actions**: Free for public repos (2,000 minutes/month)
- **Estimated usage**: ~160 minutes/year (4 runs × ~40 min each)
- **Well within limits**: 1,840 minutes remaining per month

### Database

- **Neon**: Free tier (0.5 GB storage)
- Monitor storage usage in Neon dashboard
- Archive old data if approaching limit

## Next Steps After Pipeline

After the pipeline completes successfully:

1. **Verify Data**: Check database for new records
   ```sql
   SELECT COUNT(*) FROM crimes;
   SELECT MAX(occurred_at) FROM crimes;
   ```

2. **Generate Features** (if implemented):
   ```bash
   python -m feature_engineering.feature_generator
   ```

3. **Train Models** (if implemented):
   ```bash
   python -m training.trainer
   ```

## Support

For issues or questions:
1. Check logs in GitHub Actions artifacts
2. Review this documentation
3. Check [Troubleshooting Guide](../docs/troubleshooting_connection.md)
4. Review individual layer READMEs:
   - [Bronze Layer](../ingestion/bronze/README.md)
   - [Silver Layer](../ingestion/silver/README.md)
   - [Gold Layer](../transformations/gold/README.md)

