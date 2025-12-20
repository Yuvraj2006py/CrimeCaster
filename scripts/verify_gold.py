"""Verification script for Gold layer output."""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_database_connection():
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD")
        if all([host, password]):
            database_url = f"postgresql://{db_user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError("DATABASE_URL not set")
    return create_engine(database_url)


def verify_gold_layer():
    """Verify Gold layer data in database."""
    print("=" * 60)
    print("GOLD LAYER VERIFICATION")
    print("=" * 60)
    
    try:
        engine = get_database_connection()
        
        with engine.connect() as conn:
            # Check total records
            result = conn.execute(text("SELECT COUNT(*) FROM crimes"))
            total = result.fetchone()[0]
            print(f"\nTotal crimes in database: {total:,}")
            
            # Check records with H3
            result = conn.execute(
                text("SELECT COUNT(*) FROM crimes WHERE h3_index IS NOT NULL")
            )
            h3_count = result.fetchone()[0]
            print(f"Records with H3 index: {h3_count:,}")
            
            # Check records with geometry
            result = conn.execute(
                text("SELECT COUNT(*) FROM crimes WHERE geom IS NOT NULL")
            )
            geom_count = result.fetchone()[0]
            print(f"Records with PostGIS geometry: {geom_count:,}")
            
            # Check unique H3 cells
            result = conn.execute(
                text("SELECT COUNT(DISTINCT h3_index) FROM crimes")
            )
            unique_h3 = result.fetchone()[0]
            print(f"Unique H3 hexagons: {unique_h3:,}")
            
            # Check date range
            result = conn.execute(
                text("SELECT MIN(occurred_at), MAX(occurred_at) FROM crimes")
            )
            min_date, max_date = result.fetchone()
            print(f"Date range: {min_date} to {max_date}")
            
            # Check crime types
            result = conn.execute(
                text("SELECT crime_type, COUNT(*) FROM crimes GROUP BY crime_type ORDER BY COUNT(*) DESC")
            )
            print("\nCrime type distribution:")
            for row in result:
                print(f"  {row[0]}: {row[1]:,}")
            
            # Check source files
            result = conn.execute(
                text("SELECT source_file, COUNT(*) FROM crimes GROUP BY source_file")
            )
            print("\nSource files:")
            for row in result:
                print(f"  {row[0]}: {row[1]:,} records")
            
            # Validation checks
            print("\n" + "=" * 60)
            print("VALIDATION CHECKS")
            print("=" * 60)
            
            checks = {
                "Records exist": total > 0,
                "All records have H3": h3_count == total if total > 0 else False,
                "All records have geometry": geom_count == total if total > 0 else False,
                "Multiple H3 cells": unique_h3 > 1 if total > 0 else False,
            }
            
            all_passed = True
            for check, passed in checks.items():
                status = "[PASS]" if passed else "[FAIL]"
                print(f"{status}: {check}")
                if not passed:
                    all_passed = False
            
            print("\n" + "=" * 60)
            if all_passed:
                print("[PASS] ALL CHECKS PASSED - Gold layer data is valid!")
            else:
                print("[FAIL] SOME CHECKS FAILED - Review data")
            print("=" * 60)
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False
    
    return all_passed


if __name__ == "__main__":
    verify_gold_layer()

