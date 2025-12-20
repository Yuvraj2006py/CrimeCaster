# Database Schema and Migrations

This directory contains the database schema and migration files for Crime Caster.

## Files

- `schema.sql` - Complete database schema (run this first)
- `migrations/` - Version-controlled migration files
  - `001_initial_schema.sql` - Initial schema migration
  - `run_migrations.py` - Python script to run migrations

## Setup Instructions

### Option 1: Run schema.sql directly (Recommended for initial setup)

1. Go to your database provider's SQL Editor (Neon, Railway, etc.)
2. Copy and paste the contents of `schema.sql`
3. Click "Run" to execute

**Note**: Make sure PostGIS extension is enabled first:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

### Option 2: Use migration runner

```bash
# Make sure your .env file is configured
python sql/migrations/run_migrations.py
```

Or using Docker:

```bash
docker-compose run --rm api python sql/migrations/run_migrations.py
```

## Schema Overview

### Tables

1. **crimes** - Raw crime data from Toronto Open Data
   - Includes PostGIS geometry column
   - Indexed by H3 hexagon, timestamp, location

2. **ingestion_metadata** - Tracks CSV file ingestion
   - Supports incremental loading
   - Tracks last timestamp per file

3. **features** - Engineered features for ML
   - Temporal features (hour, day, weekend, etc.)
   - Historical activity features
   - Target variables for training

4. **model_metadata** - ML model versions and metrics
   - Stores evaluation metrics as JSONB
   - Tracks active model
   - Stores artifact paths

### Indexes

All tables have appropriate indexes for:
- Spatial queries (PostGIS GIST indexes)
- Time-based queries
- H3 hexagon lookups
- Common join patterns

### Functions and Triggers

- Automatic geometry creation from lat/lon
- Ensures only one active model at a time

## Verification

After running the schema, verify PostGIS is working:

```sql
SELECT PostGIS_version();
```

Check tables exist:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

