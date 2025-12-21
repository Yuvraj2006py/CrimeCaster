"""Add dataset_type column to crimes table if it doesn't exist."""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    conn.execute(text("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = 'crimes' 
                AND column_name = 'dataset_type'
            ) THEN
                ALTER TABLE crimes ADD COLUMN dataset_type VARCHAR(100);
                CREATE INDEX IF NOT EXISTS idx_crimes_dataset_type ON crimes(dataset_type);
                CREATE INDEX IF NOT EXISTS idx_crimes_dataset_time ON crimes(dataset_type, occurred_at);
            END IF;
        END $$;
    """))
    conn.commit()
    print("Successfully added dataset_type column (if it didn't exist)")

