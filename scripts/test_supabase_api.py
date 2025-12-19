"""
Test Supabase connection using REST API instead of direct database connection.
This works around IPv6 connectivity issues on Windows.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()


def test_supabase_api():
    """Test Supabase connection using REST API."""
    print("=" * 60)
    print("Supabase REST API Connection Tester")
    print("=" * 60)
    print()
    
    # Get API credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url:
        print("[ERROR] SUPABASE_URL not found in .env file")
        print("Please add: SUPABASE_URL=https://hibjmylxyfhcizjtmspi.supabase.co")
        sys.exit(1)
    
    if not supabase_key:
        print("[ERROR] SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY not found in .env file")
        print("Please add one of these:")
        print("  SUPABASE_ANON_KEY=your-anon-key (for client-side, respects RLS)")
        print("  SUPABASE_SERVICE_ROLE_KEY=your-service-role-key (for server-side, bypasses RLS)")
        print()
        print("Get your API keys from:")
        print("  https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/api")
        sys.exit(1)
    
    print(f"Supabase URL: {supabase_url}")
    print(f"Using key: {'ANON_KEY' if os.getenv('SUPABASE_ANON_KEY') else 'SERVICE_ROLE_KEY'}")
    print()
    
    try:
        # Create Supabase client
        print("Creating Supabase client...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("[OK] Client created successfully")
        print()
        
        # Test connection by querying a table
        print("Testing connection by querying database...")
        
        # Try to query information_schema to verify connection
        # Note: Supabase REST API doesn't directly expose information_schema,
        # so we'll try to query one of your tables instead
        
        # First, try to get table list (if we have access to a metadata table)
        # For now, let's try a simple query to verify the connection works
        try:
            # Try querying a common table that might exist
            # This will fail if tables don't exist, but that's OK - it means connection works
            result = supabase.table("crimes").select("count", count="exact").limit(0).execute()
            print("[OK] Successfully connected to Supabase!")
            print(f"[OK] Can query 'crimes' table")
            print()
            print("=" * 60)
            print("[OK] Supabase REST API connection test passed!")
            print("=" * 60)
            print()
            print("Your Supabase connection is working via REST API.")
            print("You can now use Supabase client in your application.")
            return True
        except Exception as table_error:
            # If table doesn't exist, that's fine - connection still works
            error_msg = str(table_error)
            if "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
                print("[OK] Successfully connected to Supabase!")
                print("[WARN] Tables may not exist yet (this is OK - connection works)")
                print()
                print("=" * 60)
                print("[OK] Supabase REST API connection test passed!")
                print("=" * 60)
                print()
                print("Your Supabase connection is working via REST API.")
                print("You may need to run migrations to create tables.")
                return True
            else:
                # Some other error - might be auth or connection issue
                raise
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure SUPABASE_URL is correct in your .env file")
        print("2. Make sure SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY is set")
        print("3. Get your API keys from:")
        print("   https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/api")
        print("4. Make sure your project is not paused")
        return False


if __name__ == "__main__":
    success = test_supabase_api()
    sys.exit(0 if success else 1)

