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
        
        # Check if using pgbouncer (connection pooling) - warn but allow
        if 'pgbouncer' in database_url or ':6543' in database_url:
            print("[WARN] Connection pooling URL detected!")
            print("Connection pooling may not work for migrations. Direct connection (port 5432) is recommended.")
        
        connect_args = {'connect_timeout': 10}
        
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
            print("For direct database connections, you need to use the direct connection string (port 5432).")
            print("Get it from your database provider's dashboard (Neon, Railway, etc.)")
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
    # Warn but allow it (some providers use pooling)
    if database_url and ('pgbouncer' in database_url.lower() or ':6543' in database_url):
        print("[WARN] Connection pooling URL detected. This may work, but direct connection is recommended.")
    
    # If DATABASE_URL not set, try constructing from individual variables
    if not database_url:
        print("DATABASE_URL not set, constructing from individual variables...")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD")
        
        if not all([host, password]):
            print("[ERROR] Missing required environment variables:")
            if not host:
                print("  - DB_HOST")
            if not password:
                print("  - DB_PASSWORD")
            print()
            print("Please set DATABASE_URL in your .env file with the connection string from your database provider.")
            print("For Neon: https://console.neon.tech")
            sys.exit(1)
        
        database_url = f"postgresql://{db_user}:{password}@{host}:{port}/{db_name}"
    
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
        print("2. Get the exact connection string from your database provider:")
        print("   - Neon: https://console.neon.tech → Connection Details")
        print("   - Railway: Railway dashboard → PostgreSQL → Connect")
        print("   - Render: Render dashboard → PostgreSQL → Internal Database URL")
        print("3. Copy the full connection string (including ?sslmode=require if present)")
        print("4. Paste it as DATABASE_URL in your .env file")
        sys.exit(1)


if __name__ == "__main__":
    main()
