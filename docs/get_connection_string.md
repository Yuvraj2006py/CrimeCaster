# How to Get Your Supabase Connection String

## Step-by-Step Instructions

### 1. Go to Your Supabase Project Settings

**Direct Link:**
https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database

Or navigate manually:
1. Go to https://supabase.com/dashboard
2. Click on your project: **hibjmylxyfhcizjtmspi**
3. Click **Settings** (gear icon) in the left sidebar
4. Click **Database**

### 2. Find the Connection String Section

Scroll down to the **"Connection string"** section on the Database settings page.

### 3. Select the URI Tab

You'll see two tabs:
- **URI** ← **USE THIS ONE** (for direct connection)
- **Connection pooling** (don't use this for migrations/scripts)

Click on the **"URI"** tab.

### 4. Copy the Connection String

You'll see a connection string that looks like:
```
postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
```

**Important:** 
- Click the **copy icon** (📋) next to the connection string
- Or select and copy the entire string
- Make sure you copy the **entire** string including `postgresql://` at the start

### 5. Update Your .env File

1. Open your `.env` file in the project root
2. Find the line that says:
   ```
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@[HOST]:5432/postgres
   ```
3. Replace the entire value after `=` with the connection string you copied
4. It should look like:
   ```
   DATABASE_URL=postgresql://postgres:your-actual-password@db.hibjmylxyfhcizjtmspi.supabase.co:5432/postgres
   ```
5. **Save the file**

### 6. Test the Connection

Run this command to test:
```bash
python scripts/test_connection.py
```

If successful, you'll see:
```
✅ Connection successful!
✅ PostGIS enabled!
✅ Found tables: crimes, features, ingestion_metadata, model_metadata
```

## Troubleshooting

### If you can't find the connection string:
- Make sure you're in **Settings → Database** (not API or other settings)
- Scroll down - it's below the database password section

### If the connection still fails:
- Make sure you copied the **entire** connection string
- Check that your project is not paused (if Free Tier)
- Try using the connection string exactly as shown (don't modify it)

### If you see "could not translate host name":
- The connection string from Supabase should have the correct hostname
- Make sure you're using the **URI** tab, not Connection Pooling
- Try copying the connection string again

## Visual Guide

```
Supabase Dashboard
  └─ Project: hibjmylxyfhcizjtmspi
      └─ Settings (⚙️)
          └─ Database
              └─ Scroll down to "Connection string"
                  └─ Click "URI" tab
                      └─ Copy the connection string
```

## Quick Link

**Direct link to Database Settings:**
https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database

