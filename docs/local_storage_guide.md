# Local Storage Guide - ML Training with Parquet Files

## Overview

Crime Caster now supports **local-only storage mode** for ML training, allowing you to store unlimited historical data without database size constraints.

## Storage Modes

### Mode 1: Database + Parquet (Default)
- **Database**: Stores data for API serving (limited by database size)
- **Parquet Files**: Stores all data for ML training (unlimited)
- **Use Case**: Production deployment with API + ML training

### Mode 2: Local-Only (Parquet Files)
- **Database**: Disabled
- **Parquet Files**: Stores all data for ML training (unlimited)
- **Use Case**: ML training with large historical datasets

## Configuration

Set in your `.env` file:

```bash
# Enable database loading (default: true)
LOAD_TO_DATABASE=true

# Disable database loading (local-only mode)
LOAD_TO_DATABASE=false
```

## How It Works

### Data Flow

```
Bronze Layer → Silver Layer → Gold Layer
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Parquet Files (always)          Database (if enabled)
                    ↓                                   ↓
            ML Training (unlimited)         API Serving (limited)
```

### Parquet File Location

Parquet files are saved to:
```
data/gold/parquet/
├── major-crime-indicators_20251223_133648.parquet
├── shootings-firearm-discharges_20251223_133648.parquet
└── ...
```

## Loading Data for ML Training

### Basic Usage

```python
from pathlib import Path
from training.data_loader import load_training_data
from datetime import datetime, timedelta

# Load all data
project_root = Path(".")
df = load_training_data(project_root)

# Load last 2 years
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)

df = load_training_data(
    data_dir=project_root,
    start_date=start_date,
    end_date=end_date,
)

# Filter by dataset type
df = load_training_data(
    data_dir=project_root,
    dataset_types=['major-crime-indicators', 'shootings-firearm-discharges'],
)
```

### Get Data Statistics

```python
from training.data_loader import get_data_statistics

stats = get_data_statistics(Path("."))
print(f"Total files: {stats['total_files']}")
print(f"Total rows: {stats['total_rows']:,}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Date range: {stats['date_range']}")
```

## Benefits

### 1. Unlimited Storage
- No database size limits
- Store years of historical data
- Perfect for large-scale ML training

### 2. Fast Training
- Direct file access (no SQL queries)
- Columnar format (Parquet) is optimized for analytics
- Parallel file reading

### 3. Better Compression
- Parquet files are ~70% smaller than CSV
- Snappy compression (fast decompression)
- Reduced storage costs

### 4. Easy Versioning
- Track data snapshots
- Easy to archive old data
- Simple backup/restore

## Storage Comparison

| Approach | Storage Limit | Speed | Use Case |
|----------|--------------|-------|----------|
| Database only | 512 MB (Neon free) | Medium | Small datasets |
| Parquet only | Unlimited | Fast | ML training |
| Hybrid (recommended) | Unlimited local + small DB | Fast training + fast serving | Production |

## Migration Guide

### Switching to Local-Only Mode

1. **Set environment variable:**
   ```bash
   # In .env file
   LOAD_TO_DATABASE=false
   ```

2. **Run pipeline:**
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Data will be saved to:**
   - `data/gold/parquet/*.parquet` (always)
   - Database (skipped)

### Switching Back to Database Mode

1. **Set environment variable:**
   ```bash
   # In .env file
   LOAD_TO_DATABASE=true
   ```

2. **Run pipeline:**
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Data will be saved to:**
   - `data/gold/parquet/*.parquet` (always)
   - Database (enabled)

## Example: Training Model with Local Data

```python
# training/train_model.py

from pathlib import Path
from datetime import datetime, timedelta
from training.data_loader import load_training_data
import pandas as pd

# Load training data
project_root = Path(__file__).parent.parent

# Load last 2 years for training
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)

df = load_training_data(
    data_dir=project_root,
    start_date=start_date,
    end_date=end_date,
)

print(f"Loaded {len(df):,} records for training")
print(f"Date range: {df['occurred_at'].min()} to {df['occurred_at'].max()}")
print(f"Columns: {list(df.columns)}")

# Now proceed with feature engineering and model training
# No database needed!
```

## Troubleshooting

### "Parquet directory not found"

**Solution:** Run the Gold layer first to generate Parquet files:
```bash
python -m transformations.gold.h3_mapper --process-all
```

### "pyarrow not installed"

**Solution:** Install pyarrow:
```bash
pip install pyarrow
# Or
pip install -e .
```

### "No Parquet files found"

**Solution:** 
1. Check that Gold layer completed successfully
2. Verify files exist in `data/gold/parquet/`
3. Check file permissions

## Performance Tips

1. **Use date filters** to load only needed data
2. **Use dataset_types filter** to load specific datasets
3. **Use max_files parameter** for testing (loads first N files)
4. **Parquet files are cached** - subsequent loads are faster

## Next Steps

After loading data:
1. **Feature Engineering** - Create ML features from loaded DataFrame
2. **Model Training** - Train models on local data
3. **Model Evaluation** - Evaluate model performance
4. **Model Deployment** - Deploy trained models to API

