# What's New: Local Parquet Storage for ML Training

## 🎉 Summary

Crime Caster now supports **local Parquet file storage** for ML training, allowing you to store unlimited historical data without database size constraints. This solves the 512 MB Neon free tier limitation while providing faster training performance.

## ✨ New Features

### 1. **Parquet File Storage**
- All processed data is automatically saved to Parquet files
- Location: `data/gold/parquet/*.parquet`
- ~70% smaller than CSV files
- Fast columnar format optimized for ML training

### 2. **Optional Database Loading**
- New environment variable: `LOAD_TO_DATABASE`
- Set to `false` for local-only mode (no database needed)
- Set to `true` for hybrid mode (database + Parquet)

### 3. **Local Data Loader**
- New module: `training/data_loader.py`
- Load training data directly from Parquet files
- Supports date filtering, dataset filtering
- Get data statistics without loading all files

## 📁 New Files

1. **`training/data_loader.py`**
   - `load_training_data()` - Load data from Parquet files
   - `get_data_statistics()` - Get data stats without loading

2. **`docs/local_storage_guide.md`**
   - Complete guide for using local storage
   - Examples and troubleshooting

## 🔧 Modified Files

1. **`transformations/gold/h3_mapper.py`**
   - Added `save_to_parquet()` function
   - Modified `process_single_file()` to always save Parquet
   - Made database insertion optional based on `LOAD_TO_DATABASE`
   - Updated `main()` to handle local-only mode

2. **`pyproject.toml`**
   - Added `pyarrow>=14.0.0` dependency (for Parquet support)

3. **`scripts/run_pipeline.py`**
   - Updated to show storage mode information
   - Displays whether database or local-only mode is active

## 🚀 How to Use

### Enable Local-Only Mode

1. **Set environment variable in `.env`:**
   ```bash
   LOAD_TO_DATABASE=false
   ```

2. **Run pipeline:**
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Data will be saved to:**
   - ✅ `data/gold/parquet/*.parquet` (always saved)
   - ❌ Database (skipped)

### Load Data for Training

```python
from pathlib import Path
from training.data_loader import load_training_data
from datetime import datetime, timedelta

# Load all data
df = load_training_data(Path("."))

# Load last 2 years
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)
df = load_training_data(Path("."), start_date=start_date, end_date=end_date)
```

### Check Available Data

```python
from training.data_loader import get_data_statistics

stats = get_data_statistics(Path("."))
print(f"Total rows: {stats['total_rows']:,}")
print(f"Total size: {stats['total_size_mb']} MB")
```

## 📊 Benefits

| Feature | Before | After |
|---------|--------|-------|
| **Storage Limit** | 512 MB (Neon free tier) | Unlimited |
| **Training Data** | Limited by DB size | All historical data |
| **Training Speed** | SQL queries | Direct file access (faster) |
| **File Size** | CSV (larger) | Parquet (~70% smaller) |
| **Database Required** | Yes | Optional |

## 🔄 Migration

### If You Already Have Data in Database

1. **Keep existing database data** (for API serving)
2. **Run pipeline with `LOAD_TO_DATABASE=false`** to generate Parquet files
3. **Use Parquet files for ML training** (unlimited data)
4. **Use database for API** (smaller, recent data)

### Fresh Start

1. Set `LOAD_TO_DATABASE=false` in `.env`
2. Run pipeline - all data goes to Parquet files
3. Train models using `training/data_loader.py`

## 📝 Environment Variables

Add to your `.env` file:

```bash
# Data storage mode
# true = Save to database + Parquet (hybrid mode)
# false = Save to Parquet only (local-only mode)
LOAD_TO_DATABASE=false
```

## 🎯 Use Cases

### Use Local-Only Mode When:
- ✅ Training ML models with large historical datasets
- ✅ Database size limits are a concern
- ✅ You don't need database for API serving yet
- ✅ You want faster training data loading

### Use Hybrid Mode When:
- ✅ You need both ML training AND API serving
- ✅ You want recent data in database for fast API queries
- ✅ You have database storage available

## 📚 Documentation

- **Full Guide**: See `docs/local_storage_guide.md`
- **API Reference**: See `training/data_loader.py` docstrings
- **Examples**: See guide for code examples

## ⚠️ Important Notes

1. **Parquet files are always saved** - regardless of `LOAD_TO_DATABASE` setting
2. **Database is optional** - set `LOAD_TO_DATABASE=false` to skip
3. **Install pyarrow** - Required dependency: `pip install pyarrow`
4. **File location** - Parquet files in `data/gold/parquet/`

## 🐛 Troubleshooting

### "pyarrow not installed"
```bash
pip install pyarrow
# Or
pip install -e .
```

### "Parquet directory not found"
Run the Gold layer first:
```bash
python -m transformations.gold.h3_mapper --process-all
```

### "No Parquet files found"
1. Check Gold layer completed successfully
2. Verify files in `data/gold/parquet/`
3. Check file permissions

## 🎓 Next Steps

1. **Set `LOAD_TO_DATABASE=false`** in your `.env`
2. **Run the pipeline** to generate Parquet files
3. **Use `training/data_loader.py`** to load data for ML training
4. **Train your models** with unlimited historical data!

---

**Questions?** See `docs/local_storage_guide.md` for detailed documentation.

