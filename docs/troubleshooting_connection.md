# Troubleshooting Database Connection Issues

## Common Issues and Solutions

### Issue: "could not translate host name" DNS Error

**Symptoms:**
- `psycopg2.OperationalError: could not translate host name`
- DNS resolution fails on Windows

**Causes:**
1. IPv6 connectivity issues (Supabase may resolve to IPv6 only)
2. Project might be paused (Free Tier)
3. Incorrect hostname format

**Solutions:**

#### Solution 1: Use Connection String from Supabase Dashboard (Recommended)

1. Go to: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database
2. Scroll to "Connection string" section
3. Copy the **exact** connection string from the "URI" tab
4. Paste it directly into your `.env` file as `DATABASE_URL`

The connection string from Supabase may use a different hostname format that works better.

#### Solution 2: Check if Project is Paused

1. Go to: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi
2. If you see "PAUSED", click "Restore Project"
3. Wait a few minutes for it to resume

#### Solution 3: Use Connection Pooling (IPv4 Support)

1. Go to: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database
2. Find "Connection Pooling" section
3. Copy the "Transaction Mode" connection string (port 6543)
4. Use this for your `DATABASE_URL` (but note: pooling may not work for migrations)

#### Solution 4: Use Supabase SQL Editor (Easiest for Initial Setup)

For initial schema setup, use the SQL Editor directly:
1. Go to: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/sql/new
2. Copy and paste `sql/schema.sql`
3. Run it directly

This avoids all connection string issues.

### Issue: Password Still Shows as Placeholder

**Solution:**
- Make sure you updated BOTH:
  - `SUPABASE_DB_PASSWORD=your-actual-password`
  - `DATABASE_URL=postgresql://postgres:your-actual-password@...`

### Testing Connection

```python
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()
database_url = os.getenv("DATABASE_URL")

if not database_url:
    # Construct from individual variables
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    database_url = f"postgresql://postgres:{password}@{host}:{port}/{db_name}"

try:
    engine = create_engine(database_url, connect_args={'connect_timeout': 10})
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print("✅ Connection successful!")
        print("PostgreSQL version:", result.fetchone()[0][:50])
except Exception as e:
    print("❌ Connection failed:", e)
```

