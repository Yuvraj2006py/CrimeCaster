# Neon Database Setup Guide

This guide will help you set up Neon (PostgreSQL with PostGIS) for CrimeCaster.

## Why Neon?

- ✅ **Free tier**: 0.5 GB storage, unlimited projects
- ✅ **PostGIS support**: Built-in spatial database extension
- ✅ **No pausing**: Unlike Supabase free tier
- ✅ **Easy setup**: Simple connection string
- ✅ **Great performance**: Serverless PostgreSQL

## Step 1: Create Neon Account

1. Go to: https://neon.tech
2. Click **"Sign Up"** (free)
3. Sign up with GitHub, Google, or email

## Step 2: Create a Project

1. After signing in, click **"Create a project"**
2. Fill in:
   - **Project name**: `crime-caster` (or any name)
   - **Region**: Choose closest to you (e.g., `us-east-2`)
   - **PostgreSQL version**: `15` or `16` (both work)
3. Click **"Create project"**

## Step 3: Enable PostGIS Extension

1. In your Neon project dashboard, click **"SQL Editor"** in the left sidebar
2. Click **"New query"**
3. Run this command:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
4. Click **"Run"** (or press Ctrl+Enter)
5. You should see: `Success. No rows returned`

## Step 4: Get Connection String

1. In Neon dashboard, click **"Connection Details"** (or "Connection string")
2. You'll see a connection string like:
   ```
   postgresql://username:password@ep-xxxxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
3. Click the **copy icon** to copy the full connection string

**Important**: Copy the entire string including `?sslmode=require` at the end.

## Step 5: Update .env File

1. Open your `.env` file in the project root
2. Add or update the `DATABASE_URL`:

```env
# Neon Database Connection
DATABASE_URL=postgresql://username:password@ep-xxxxx.us-east-2.aws.neon.tech/neondb?sslmode=require
```

**Replace** the connection string with the one you copied from Neon.

3. **Remove** old Supabase variables (if present):
   ```env
   # Remove these (no longer needed):
   # SUPABASE_URL=...
   # SUPABASE_DB_HOST=...
   # SUPABASE_DB_PORT=...
   # SUPABASE_DB_NAME=...
   # SUPABASE_DB_PASSWORD=...
   ```

4. **Save** the `.env` file

## Step 6: Run Database Schema

You have two options:

### Option A: Use Neon SQL Editor (Recommended)

1. Go to Neon dashboard → **SQL Editor**
2. Click **"New query"**
3. Open `sql/schema.sql` from your project
4. Copy the entire contents
5. Paste into Neon SQL Editor
6. Click **"Run"**
7. Wait for completion (should see "Success")

### Option B: Use Migration Script

```bash
python sql/migrations/run_migrations.py
```

## Step 7: Test Connection

```bash
python scripts/test_connection.py
```

You should see:
```
[OK] Connection successful!
[OK] PostGIS enabled!
[OK] Found tables: crimes, features, ingestion_metadata, model_metadata
```

## Step 8: Load Data

Now you can run the Gold layer to load data:

```bash
python -m transformations.gold.h3_mapper
```

This will:
1. Load cleaned data from Silver layer
2. Map to H3 hexagons
3. Insert into Neon database
4. Take about 5-10 minutes for ~388K records

## Verification

After loading data, verify it worked:

```bash
python scripts/verify_gold.py
```

Or check in Neon SQL Editor:
```sql
SELECT COUNT(*) FROM crimes;
-- Should show ~388,516 records
```

## Troubleshooting

### "Connection refused" or "Could not connect"

1. **Check connection string**: Make sure you copied the entire string from Neon
2. **Check SSL mode**: Neon requires SSL, make sure `?sslmode=require` is in the URL
3. **Check project status**: Make sure your Neon project is active (not paused)

### "Extension postgis does not exist"

1. Go to Neon SQL Editor
2. Run: `CREATE EXTENSION IF NOT EXISTS postgis;`
3. Wait a few seconds
4. Try again

### "Table does not exist"

1. Run the schema: `python sql/migrations/run_migrations.py`
2. Or use Neon SQL Editor to run `sql/schema.sql`

### Connection timeout

- Neon uses serverless architecture, first connection may take a few seconds
- This is normal - subsequent connections are faster

## Neon Dashboard Features

- **SQL Editor**: Run queries directly in browser
- **Connection Details**: Get connection strings
- **Branches**: Create database branches (like git branches)
- **Metrics**: Monitor database usage
- **Settings**: Configure database settings

## Free Tier Limits

- **Storage**: 0.5 GB (enough for ~500K crime records)
- **Compute**: 0.5 vCPU (sufficient for this project)
- **Projects**: Unlimited
- **No time limits**: Unlike Supabase, Neon doesn't pause projects

## Upgrading (if needed)

If you need more storage/compute:
1. Go to Neon dashboard → **Settings** → **Billing**
2. Choose a plan (starts at $19/month)
3. Your connection string stays the same

## Next Steps

After setup:
1. ✅ Database connected
2. ✅ Schema created
3. ✅ PostGIS enabled
4. ✅ Ready to load data with Gold layer
5. ✅ Ready for feature engineering and ML training

## Support

- Neon Docs: https://neon.tech/docs
- Neon Discord: https://discord.gg/neondatabase
- Neon Status: https://status.neon.tech

