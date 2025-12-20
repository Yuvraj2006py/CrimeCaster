# Quick Neon Setup Guide

## 🚀 5-Minute Setup

### Step 1: Create Neon Account
1. Go to: https://neon.tech
2. Click **"Sign Up"** (free, no credit card needed)
3. Sign up with GitHub, Google, or email

### Step 2: Create Project
1. Click **"Create a project"**
2. Name: `crime-caster` (or any name)
3. Region: Choose closest to you
4. PostgreSQL: `15` or `16` (both work)
5. Click **"Create project"**

### Step 3: Enable PostGIS
1. In Neon dashboard, click **"SQL Editor"**
2. Click **"New query"**
3. Paste and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
4. Click **"Run"** ✅

### Step 4: Get Connection String
1. Click **"Connection Details"** (or "Connection string")
2. Copy the connection string (looks like):
   ```
   postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

### Step 5: Update .env File
1. Open `.env` file in project root
2. Add/update:
   ```env
   DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
   (Use the connection string you copied from Neon)
3. **Remove** old Supabase variables (if any):
   ```env
   # Remove these:
   # SUPABASE_URL=...
   # SUPABASE_DB_HOST=...
   # etc.
   ```
4. Save file

### Step 6: Run Schema
**Option A: Neon SQL Editor (Easiest)**
1. Go to Neon → SQL Editor
2. Open `sql/schema.sql` from your project
3. Copy entire contents
4. Paste in Neon SQL Editor
5. Click **"Run"**

**Option B: Migration Script**
```bash
python sql/migrations/run_migrations.py
```

### Step 7: Test Connection
```bash
python scripts/test_connection.py
```

Should see:
```
[OK] Connection successful!
[OK] PostGIS enabled!
[OK] Found tables: crimes, features, ingestion_metadata, model_metadata
```

### Step 8: Load Data
```bash
python -m transformations.gold.h3_mapper
```

This loads ~388K crime records into your database.

## ✅ Done!

Your database is now set up and ready to use.

## Troubleshooting

**"Connection refused"**
- Check connection string is correct
- Make sure you copied the entire string including `?sslmode=require`

**"Extension postgis does not exist"**
- Run: `CREATE EXTENSION IF NOT EXISTS postgis;` in Neon SQL Editor

**"Table does not exist"**
- Run schema: `python sql/migrations/run_migrations.py`

## Need Help?

- Full guide: See `docs/neon_setup.md`
- Neon docs: https://neon.tech/docs
- Test connection: `python scripts/test_connection.py`

