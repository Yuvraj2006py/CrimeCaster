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
        print(f"URL (masked): {database_url.split('@')[0]}@[HOST]/[DB]")
        
        engine = create_engine(database_url, connect_args={'connect_timeout': 10})
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connection successful!")
            print(f"PostgreSQL version: {version[:60]}...")
            
            # Test PostGIS
            result = conn.execute(text("SELECT PostGIS_version()"))
            postgis_version = result.fetchone()[0]
            print(f"✅ PostGIS enabled! Version: {postgis_version}")
            
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
                print(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
            else:
                print("⚠️  No tables found yet (schema may not be created)")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Database Connection Tester")
    print("=" * 60)
    print()
    
    # Try DATABASE_URL first
    database_url = os.getenv("DATABASE_URL")
    
    # If not set, try constructing from individual variables
    if not database_url:
        print("DATABASE_URL not set, constructing from individual variables...")
        host = os.getenv("SUPABASE_DB_HOST")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        
        if not all([host, password]):
            print("❌ Missing required environment variables:")
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
        print("✅ Connection test passed!")
        print("=" * 60)
        print()
        print("Your database connection is working correctly.")
        print("You can now run migrations and use the database.")
    else:
        print()
        print("=" * 60)
        print("❌ Connection test failed")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("1. Make sure DATABASE_URL is set in your .env file")
        print("2. Get the exact connection string from Supabase Dashboard:")
        print("   https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database")
        print("3. Copy the 'URI' connection string (not pooling)")
        print("4. Paste it as DATABASE_URL in your .env file")
        sys.exit(1)


if __name__ == "__main__":
    main()

