# Database Setup Guide

This project uses **PostgreSQL with PostGIS** for spatial data storage.

## Supported Providers

We support multiple PostgreSQL providers:

- ✅ **Neon** (Recommended) - Free tier, PostGIS, no pausing
- ✅ **Railway** - Easy setup, PostGIS support
- ✅ **Render** - Free tier, PostGIS support
- ✅ **Local Docker** - Full control, free
- ✅ **AWS RDS** - Production-ready
- ✅ **Any PostgreSQL** - Standard PostgreSQL with PostGIS

## Quick Setup

### Using Neon (Recommended)

See [Neon Setup Guide](neon_setup.md) for detailed instructions.

**Quick steps:**
1. Sign up at https://neon.tech
2. Create project → Enable PostGIS
3. Copy connection string
4. Update `.env`: `DATABASE_URL=postgresql://...`
5. Run schema: `python sql/migrations/run_migrations.py`

### Using Local Docker

Add to `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: crimecaster
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Then:
```bash
docker-compose up -d postgres
# Wait 10 seconds for startup
python sql/migrations/run_migrations.py
```

## Environment Variables

### Required

```env
DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=require
```

### Optional (for fallback)

If `DATABASE_URL` is not set, code will try to construct from:
```env
DB_HOST=your-host
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
```

## Schema Setup

### Option 1: SQL Editor (Easiest)

1. Open SQL editor in your database provider
2. Copy contents of `sql/schema.sql`
3. Paste and run

### Option 2: Migration Script

```bash
python sql/migrations/run_migrations.py
```

## Verify Setup

```bash
# Test connection
python scripts/test_connection.py

# Verify schema
python scripts/verify_gold.py
```

## Requirements

- **PostgreSQL**: 12+ (15+ recommended)
- **PostGIS**: 3.0+ (required for spatial features)
- **Extensions**: PostGIS must be enabled

## Connection String Format

Standard PostgreSQL connection string:
```
postgresql://[user]:[password]@[host]:[port]/[database]?sslmode=require
```

Examples:
- Neon: `postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require`
- Railway: `postgresql://user:pass@containers-us-west-xxx.railway.app:5432/railway?sslmode=require`
- Local: `postgresql://postgres:postgres@localhost:5432/crimecaster`

## Troubleshooting

### "Extension postgis does not exist"

Run in SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

### "Connection refused"

1. Check connection string is correct
2. Check database is running/active
3. Check firewall/network settings
4. For Neon: Ensure `?sslmode=require` is in URL

### "Table does not exist"

Run schema:
```bash
python sql/migrations/run_migrations.py
```

## Migration from Supabase

If migrating from Supabase:

1. Remove Supabase-specific env vars from `.env`
2. Add `DATABASE_URL` with new provider connection string
3. Run schema in new database
4. Test connection: `python scripts/test_connection.py`
5. Load data: `python -m transformations.gold.h3_mapper`

## Next Steps

After database is set up:
1. ✅ Run Bronze layer: `python -m ingestion.bronze.csv_loader`
2. ✅ Run Silver layer: `python -m ingestion.silver.cleaner`
3. ✅ Run Gold layer: `python -m transformations.gold.h3_mapper`
4. ✅ Verify: `python scripts/verify_gold.py`

