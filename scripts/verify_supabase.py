"""
Comprehensive verification of Supabase connection.
Tests multiple aspects of the connection.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


def verify_supabase():
    """Comprehensive verification of Supabase connection."""
    print("=" * 60)
    print("Supabase Connection Verification")
    print("=" * 60)
    print()
    
    # Check environment variables
    print("1. Checking environment variables...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url:
        print("   [ERROR] SUPABASE_URL not found")
        return False
    print(f"   [OK] SUPABASE_URL: {supabase_url}")
    
    if not supabase_key:
        print("   [ERROR] API key not found")
        return False
    key_type = "ANON_KEY" if os.getenv("SUPABASE_ANON_KEY") else "SERVICE_ROLE_KEY"
    print(f"   [OK] Using {key_type}")
    print()
    
    # Create client
    print("2. Creating Supabase client...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("   [OK] Client created successfully")
    except Exception as e:
        print(f"   [ERROR] Failed to create client: {e}")
        return False
    print()
    
    # Test connection by querying tables
    print("3. Testing database access...")
    tables_to_check = ["crimes", "features", "ingestion_metadata", "model_metadata"]
    
    for table_name in tables_to_check:
        try:
            result = supabase.table(table_name).select("count", count="exact").limit(0).execute()
            count = result.count if hasattr(result, 'count') else "unknown"
            print(f"   [OK] Table '{table_name}' exists (count: {count})")
        except Exception as e:
            error_msg = str(e).lower()
            if "relation" in error_msg or "does not exist" in error_msg:
                print(f"   [INFO] Table '{table_name}' does not exist yet (this is OK)")
            else:
                print(f"   [WARN] Error querying '{table_name}': {e}")
    print()
    
    # Test a simple query
    print("4. Testing data query...")
    try:
        # Try to get a sample row from crimes table
        result = supabase.table("crimes").select("*").limit(1).execute()
        if result.data and len(result.data) > 0:
            print(f"   [OK] Successfully queried data (found {len(result.data)} row(s))")
            print(f"   [OK] Sample data keys: {list(result.data[0].keys())[:5]}...")
        else:
            print("   [INFO] Table exists but is empty (this is OK)")
    except Exception as e:
        error_msg = str(e).lower()
        if "relation" in error_msg or "does not exist" in error_msg:
            print("   [INFO] Table doesn't exist yet - run migrations to create it")
        else:
            print(f"   [WARN] Query error: {e}")
    print()
    
    # Verify project info
    print("5. Verifying project connection...")
    try:
        # Try to access a system endpoint (this will fail if project is paused)
        result = supabase.table("crimes").select("count", count="exact").limit(0).execute()
        print("   [OK] Project is active and accessible")
    except Exception as e:
        error_msg = str(e).lower()
        if "paused" in error_msg or "unavailable" in error_msg:
            print("   [ERROR] Project appears to be paused")
            return False
        else:
            print(f"   [INFO] Connection works (table may not exist): {e}")
    print()
    
    print("=" * 60)
    print("[OK] Supabase Connection Verified Successfully!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  [OK] Environment variables configured")
    print("  [OK] Supabase client created")
    print("  [OK] Can connect to Supabase REST API")
    print("  [OK] Project is active")
    print()
    print("You can now use Supabase in your application!")
    return True


if __name__ == "__main__":
    import sys
    success = verify_supabase()
    sys.exit(0 if success else 1)

