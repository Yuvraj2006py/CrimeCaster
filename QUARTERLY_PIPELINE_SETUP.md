# Quarterly Pipeline Setup - Quick Start Guide

## ✅ What Was Created

1. **`scripts/run_pipeline.py`** - Unified pipeline orchestrator
2. **`.github/workflows/quarterly-pipeline.yml`** - GitHub Actions workflow
3. **`scripts/test_pipeline.py`** - Setup verification script
4. **`docs/quarterly_pipeline.md`** - Complete documentation
5. **`logs/.gitkeep`** - Logs directory tracker

## 🧪 Testing Steps

### Step 1: Verify Setup (No Database Required)

Run the test script to verify everything is set up correctly:

```bash
python scripts/test_pipeline.py
```

**Expected Output:**
```
✅ Bronze layer import successful
✅ Silver layer import successful
✅ Gold layer import successful
✅ All directories exist
✅ Pipeline script exists
✅ GitHub workflow file exists
🎉 All tests passed!
```

### Step 2: Test Locally (Requires Database)

**Before running**, make sure:
- ✅ `.env` file has `DATABASE_URL` set
- ✅ Database is accessible
- ✅ You have internet connection (for Bronze layer download)

**Run the pipeline:**

```bash
python scripts/run_pipeline.py
```

**What to expect:**
- Bronze layer downloads 9 datasets (takes 5-10 minutes)
- Silver layer cleans the data (takes 2-5 minutes)
- Gold layer loads to database (takes 5-15 minutes)
- Total: ~15-30 minutes

**Check logs:**
```bash
# View latest log
cat logs/pipeline_$(date +%Y-%m-%d).log

# Or on Windows PowerShell
Get-Content logs\pipeline_$(Get-Date -Format "yyyy-MM-dd").log
```

### Step 3: Test GitHub Actions (Manual Trigger)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add quarterly pipeline automation"
   git push origin main
   ```

2. **Set up GitHub Secret:**
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `DATABASE_URL`
   - Value: Your Neon PostgreSQL connection string
   - Click "Add secret"

3. **Test Workflow Manually:**
   - Go to: Repository → Actions tab
   - Select "Quarterly Data Pipeline"
   - Click "Run workflow" button
   - Select branch: `main`
   - Click "Run workflow"

4. **Monitor Execution:**
   - Watch the workflow run in real-time
   - Check each step for success/failure
   - Download logs artifact if needed

## 📋 Pre-Deployment Checklist

Before pushing to production:

- [ ] Test script passes: `python scripts/test_pipeline.py`
- [ ] Local pipeline test succeeds: `python scripts/run_pipeline.py`
- [ ] `DATABASE_URL` is set in GitHub Secrets
- [ ] Repository is **public** (required for free GitHub Actions)
- [ ] GitHub Actions is enabled (Settings → Actions → General)
- [ ] Workflow file is in `.github/workflows/`
- [ ] Logs directory exists and is in `.gitignore`

## 🎯 Verification Commands

### Quick Health Check
```bash
# Test imports and setup
python scripts/test_pipeline.py

# Check if workflow file exists
ls -la .github/workflows/quarterly-pipeline.yml

# Check if pipeline script exists
ls -la scripts/run_pipeline.py
```

### Database Verification (After Pipeline Runs)
```sql
-- Check total records
SELECT COUNT(*) FROM crimes;

-- Check latest ingestion
SELECT * FROM ingestion_metadata 
ORDER BY ingested_at DESC 
LIMIT 5;

-- Check date range
SELECT MIN(occurred_at), MAX(occurred_at) FROM crimes;
```

## 🐛 Troubleshooting

### Test Script Fails
- **Import errors**: Run `pip install -e .` to install dependencies
- **Directory errors**: Scripts create directories automatically
- **Workflow not found**: Make sure `.github/workflows/` directory exists

### Local Pipeline Fails
- **Bronze layer**: Check internet connection, Toronto Open Data API
- **Silver layer**: Check Bronze output exists in `data/raw/`
- **Gold layer**: Verify `DATABASE_URL` in `.env` is correct

### GitHub Actions Fails
- **Secret not found**: Verify `DATABASE_URL` is set in Secrets
- **Workflow not running**: Check repository is public
- **Timeout**: Increase `timeout-minutes` in workflow file

## 📅 Schedule Verification

The pipeline is scheduled to run:
- **January 1** at 2:00 AM UTC
- **April 1** at 2:00 AM UTC
- **July 1** at 2:00 AM UTC
- **October 1** at 2:00 AM UTC

**To verify next run date:**
1. Go to Actions → Quarterly Data Pipeline
2. Click on "Scheduled" runs
3. Check the next scheduled time

**To test schedule immediately:**
- Edit `.github/workflows/quarterly-pipeline.yml`
- Change cron to: `*/5 * * * *` (runs every 5 minutes)
- Push and wait for test run
- **Remember to change back** to quarterly schedule!

## 📊 Monitoring

### After Each Run, Check:
1. ✅ All three phases completed successfully
2. ✅ Logs artifact was uploaded
3. ✅ Database has new records
4. ✅ No errors in workflow logs

### Monthly Check:
- Review GitHub Actions usage (Settings → Billing)
- Check Neon database storage usage
- Review any failed runs

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Test script passes all checks
- ✅ Local pipeline completes successfully
- ✅ GitHub Actions workflow runs manually
- ✅ Logs show all phases completed
- ✅ Database has new crime records

## 📚 Additional Resources

- **Full Documentation**: `docs/quarterly_pipeline.md`
- **Bronze Layer**: `ingestion/bronze/README.md`
- **Silver Layer**: `ingestion/silver/README.md`
- **Gold Layer**: `transformations/gold/README.md`

## 🆘 Need Help?

1. Check logs in `logs/` directory
2. Review GitHub Actions logs
3. Run test script: `python scripts/test_pipeline.py`
4. Check documentation: `docs/quarterly_pipeline.md`

