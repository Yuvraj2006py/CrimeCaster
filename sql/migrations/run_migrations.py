"""
Database migration runner for Crime Caster.

This script runs SQL migration files in order.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_database_url():
    """Get database URL from environment variables."""
    # Prefer direct connection for migrations
    database_url = os.getenv("DATABASE_URL")
    
    # If DATABASE_URL is not set, construct it from individual Supabase variables
    if not database_url:
        host = os.getenv("SUPABASE_DB_HOST")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        
        if not host:
            raise ValueError("SUPABASE_DB_HOST not found in environment variables")
        
        if not password:
            raise ValueError("SUPABASE_DB_PASSWORD not found in environment variables")
        
        # Check for placeholder password
        if "your-database-password" in password or "your-" in password.lower():
            raise ValueError(
                "Please update SUPABASE_DB_PASSWORD in your .env file with your actual database password.\n"
                "Get it from: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database"
            )
        
        database_url = f"postgresql://postgres:{password}@{host}:{port}/{db_name}"
    
    # If DATABASE_URL uses connection pooling, use direct connection instead
    elif database_url and "pgbouncer=true" in database_url:
        # Extract components and rebuild with direct port
        host = os.getenv("SUPABASE_DB_HOST")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        
        # Check for placeholder password
        if password and ("your-database-password" in password or "your-" in password.lower()):
            raise ValueError(
                "Please update SUPABASE_DB_PASSWORD in your .env file with your actual database password.\n"
                "Get it from: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi/settings/database"
            )
        
        if not host:
            raise ValueError("SUPABASE_DB_HOST not found in environment variables")
        
        if not password:
            raise ValueError("SUPABASE_DB_PASSWORD not found in environment variables")
        
        database_url = f"postgresql://postgres:{password}@{host}:{port}/{db_name}"
    
    # Validate password is not placeholder
    if "your-database-password" in database_url or "your-" in database_url.lower():
        raise ValueError(
            "Please update your database password in .env file.\n"
            "Get it from Supabase Dashboard → Settings → Database"
        )
    
    return database_url


def run_migrations():
    """Run all migration files in order."""
    migrations_dir = Path(__file__).parent
    
    # Get all SQL migration files, sorted
    migration_files = sorted(
        migrations_dir.glob("*.sql"),
        key=lambda x: x.name
    )
    
    if not migration_files:
        logger.warning("No migration files found")
        return
    
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    logger.info(f"Found {len(migration_files)} migration files")
    
    with engine.connect() as conn:
        # Create migrations tracking table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
        
        # Get already applied migrations
        applied = {
            row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))
        }
        
        # Run migrations
        for migration_file in migration_files:
            version = migration_file.stem
            
            if version in applied:
                logger.info(f"Skipping {version} (already applied)")
                continue
            
            logger.info(f"Running migration: {version}")
            
            try:
                sql = migration_file.read_text(encoding="utf-8")
                
                # Execute migration
                conn.execute(text(sql))
                conn.commit()
                
                # Record migration
                conn.execute(
                    text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                    {"version": version}
                )
                conn.commit()
                
                logger.success(f"Successfully applied {version}")
                
            except Exception as e:
                logger.error(f"Failed to apply {version}: {e}")
                conn.rollback()
                raise
    
    logger.success("All migrations completed")


if __name__ == "__main__":
    try:
        run_migrations()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

