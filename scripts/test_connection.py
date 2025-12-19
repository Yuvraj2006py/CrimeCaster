"""
Test database connection and help configure .env file.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_connection(database_url: str):
    """Test database connection."""
    try:
        print(f"Testing connection...")
        masked_url = database_url.split('@')[0] + "@[HOST]/[DB]" if '@' in database_url else "[MASKED]"
        print(f"URL (masked): {masked_url}")
        
        # Check if using pgbouncer (connection pooling) - not supported for direct connections
        if 'pgbouncer' in database_url or ':6543' in database_url:
            print("[ERROR] Connection pooling URL detected!")
            print("You're using a connection pooling URL (pgbouncer), which doesn't work for direct connections.")
            print("Please use the direct connection URL (port 5432) instead.")
            print("Get it from: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database")
            print("Make sure to use the 'URI' tab, NOT the 'Connection pooling' tab.")
            return False
        
        # Try to handle IPv6-only connections on Windows
        connect_args = {'connect_timeout': 10}
        # Force IPv4 if possible (some Supabase instances are IPv6-only)
        try:
            import socket
            # Test if we can resolve to IPv4
            socket.getaddrinfo('db.hibjmylxyfhcizjtmspi.supabase.co', 5432, socket.AF_INET)
        except:
            # IPv4 not available, will try IPv6
            pass
        
        engine = create_engine(database_url, connect_args=connect_args)
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"[OK] Connection successful!")
            print(f"PostgreSQL version: {version[:60]}...")
            
            # Test PostGIS
            result = conn.execute(text("SELECT PostGIS_version()"))
            postgis_version = result.fetchone()[0]
            print(f"[OK] PostGIS enabled! Version: {postgis_version}")
            
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('crimes', 'features', 'ingestion_metadata', 'model_metadata')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            if tables:
                print(f"[OK] Found {len(tables)} tables: {', '.join(tables)}")
            else:
                print("[WARN] No tables found yet (schema may not be created)")
            
            return True
            
    except Exception as e:
        error_msg = str(e)
        if 'pgbouncer' in error_msg.lower():
            print(f"[ERROR] Connection failed: Invalid connection pooling URL detected")
            print("Your DATABASE_URL is using a connection pooling URL (pgbouncer).")
            print("For direct database connections, you need to use the 'URI' connection string (port 5432).")
            print("Get it from: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database")
            print("Click on the 'URI' tab (NOT 'Connection pooling') and copy that connection string.")
        else:
            print(f"[ERROR] Connection failed: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Database Connection Tester")
    print("=" * 60)
    print()
    
    # Try DATABASE_URL first
    database_url = os.getenv("DATABASE_URL")
    
    # Check if DATABASE_URL is using connection pooling (pgbouncer)
    # If so, prefer constructing from individual variables which should use port 5432
    if database_url and ('pgbouncer' in database_url.lower() or ':6543' in database_url):
        print("DATABASE_URL is using connection pooling (pgbouncer).")
        print("Constructing direct connection URL from individual variables (port 5432)...")
        host = os.getenv("SUPABASE_DB_HOST")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        
        if all([host, password]):
            database_url = f"postgresql://postgres:{password}@{host}:{port}/{db_name}"
            print(f"Using direct connection URL with port {port}")
        else:
            print("[WARN] Individual variables not complete, will try DATABASE_URL anyway")
            print("Missing variables:")
            if not host:
                print("  - SUPABASE_DB_HOST")
            if not password:
                print("  - SUPABASE_DB_PASSWORD")
    
    # If DATABASE_URL not set, try constructing from individual variables
    if not database_url:
        print("DATABASE_URL not set, constructing from individual variables...")
        host = os.getenv("SUPABASE_DB_HOST")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        
        if not all([host, password]):
            print("[ERROR] Missing required environment variables:")
            if not host:
                print("  - SUPABASE_DB_HOST")
            if not password:
                print("  - SUPABASE_DB_PASSWORD")
            print()
            print("Please set DATABASE_URL in your .env file with the connection string from Supabase.")
            print("Get it from: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database")
            sys.exit(1)
        
        database_url = f"postgresql://postgres:{password}@{host}:{port}/{db_name}"
    
    # Test connection
    success = test_connection(database_url)
    
    if success:
        print()
        print("=" * 60)
        print("[OK] Connection test passed!")
        print("=" * 60)
        print()
        print("Your database connection is working correctly.")
        print("You can now run migrations and use the database.")
    else:
        print()
        print("=" * 60)
        print("[ERROR] Connection test failed")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("1. Make sure DATABASE_URL is set in your .env file")
        print("2. Get the exact connection string from Supabase Dashboard:")
        print("   https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database")
        print("3. IMPORTANT: Click on the 'URI' tab (NOT 'Connection pooling')")
        print("4. Copy the connection string (should use port 5432, not 6543)")
        print("5. Paste it as DATABASE_URL in your .env file")
        sys.exit(1)


if __name__ == "__main__":
    main()

