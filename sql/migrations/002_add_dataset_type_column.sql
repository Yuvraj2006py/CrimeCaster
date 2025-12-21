-- Migration: 002_add_dataset_type_column.sql
-- Description: Add dataset_type column to crimes table if it doesn't exist
-- Created: 2025-12-21

-- Add dataset_type column to crimes table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'crimes' 
        AND column_name = 'dataset_type'
    ) THEN
        ALTER TABLE crimes ADD COLUMN dataset_type VARCHAR(100);
        
        -- Add index for dataset_type
        CREATE INDEX IF NOT EXISTS idx_crimes_dataset_type ON crimes(dataset_type);
        
        -- Add composite index for dataset_type and occurred_at
        CREATE INDEX IF NOT EXISTS idx_crimes_dataset_time ON crimes(dataset_type, occurred_at);
        
        -- Add comment
        COMMENT ON COLUMN crimes.dataset_type IS 'Dataset type/key: major-crime-indicators, shootings-firearm-discharges, etc.';
    END IF;
END $$;

