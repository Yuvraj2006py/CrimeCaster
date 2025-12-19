-- Crime Caster Toronto Database Schema
-- PostgreSQL + PostGIS

-- Ensure PostGIS extension is enabled (should already be done via Supabase SQL Editor)
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- CRIMES TABLE
-- Stores raw crime data from Toronto Open Data
-- ============================================================================

CREATE TABLE IF NOT EXISTS crimes (
    id BIGSERIAL PRIMARY KEY,
    crime_type VARCHAR(100) NOT NULL,
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    neighbourhood VARCHAR(100),
    premise_type VARCHAR(100),
    h3_index VARCHAR(20) NOT NULL,  -- H3 hexagon index (resolution 9)
    source_file VARCHAR(255) NOT NULL,  -- Original CSV filename
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- PostGIS geometry column for spatial queries
    geom GEOMETRY(POINT, 4326)
);

-- Indexes for crimes table
CREATE INDEX IF NOT EXISTS idx_crimes_occurred_at ON crimes(occurred_at);
CREATE INDEX IF NOT EXISTS idx_crimes_h3_index ON crimes(h3_index);
CREATE INDEX IF NOT EXISTS idx_crimes_crime_type ON crimes(crime_type);
CREATE INDEX IF NOT EXISTS idx_crimes_neighbourhood ON crimes(neighbourhood);
CREATE INDEX IF NOT EXISTS idx_crimes_source_file ON crimes(source_file);
CREATE INDEX IF NOT EXISTS idx_crimes_geom ON crimes USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_crimes_location ON crimes(latitude, longitude);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_crimes_h3_time ON crimes(h3_index, occurred_at);

-- ============================================================================
-- INGESTION_METADATA TABLE
-- Tracks CSV file ingestion for incremental loading
-- ============================================================================

CREATE TABLE IF NOT EXISTS ingestion_metadata (
    id BIGSERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL UNIQUE,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    record_count INTEGER NOT NULL DEFAULT 0,
    last_timestamp TIMESTAMP WITH TIME ZONE,  -- Last crime timestamp in this file
    status VARCHAR(50) DEFAULT 'completed',  -- completed, failed, in_progress
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_file_name ON ingestion_metadata(file_name);
CREATE INDEX IF NOT EXISTS idx_ingestion_ingested_at ON ingestion_metadata(ingested_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_last_timestamp ON ingestion_metadata(last_timestamp);

-- ============================================================================
-- FEATURES TABLE
-- Stores engineered features for ML training and inference
-- ============================================================================

CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    h3_index VARCHAR(20) NOT NULL,
    time_window TIMESTAMP WITH TIME ZONE NOT NULL,  -- 1-hour time window
    
    -- Temporal features
    hour_of_day INTEGER NOT NULL,  -- 0-23
    day_of_week INTEGER NOT NULL,  -- 0-6 (Monday=0)
    is_weekend BOOLEAN NOT NULL DEFAULT FALSE,
    is_night BOOLEAN NOT NULL DEFAULT FALSE,
    is_holiday BOOLEAN NOT NULL DEFAULT FALSE,
    month INTEGER NOT NULL,  -- 1-12
    season INTEGER NOT NULL,  -- 1-4 (Spring=1, Summer=2, Fall=3, Winter=4)
    
    -- Historical activity features
    crimes_last_1h INTEGER NOT NULL DEFAULT 0,
    crimes_last_24h INTEGER NOT NULL DEFAULT 0,
    crimes_last_7d INTEGER NOT NULL DEFAULT 0,
    same_hour_last_week INTEGER NOT NULL DEFAULT 0,
    rolling_avg_30d DECIMAL(10, 4) NOT NULL DEFAULT 0,
    
    -- Target variable (for training)
    target_crime_count INTEGER,  -- Actual crime count in this window (for training)
    target_binary INTEGER,  -- Binary: 1 if crime occurred, 0 otherwise
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one feature set per h3_index + time_window
    UNIQUE(h3_index, time_window)
);

CREATE INDEX IF NOT EXISTS idx_features_h3_index ON features(h3_index);
CREATE INDEX IF NOT EXISTS idx_features_time_window ON features(time_window);
CREATE INDEX IF NOT EXISTS idx_features_h3_time ON features(h3_index, time_window);
CREATE INDEX IF NOT EXISTS idx_features_target ON features(target_binary) WHERE target_binary IS NOT NULL;

-- ============================================================================
-- MODEL_METADATA TABLE
-- Tracks ML model versions, training metrics, and artifacts
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_metadata (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL UNIQUE,  -- e.g., 'logistic_regression_v1'
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- logistic_regression, xgboost, lightgbm, etc.
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    artifact_path VARCHAR(500) NOT NULL,  -- Path to serialized model file
    is_active BOOLEAN DEFAULT FALSE,  -- Only one active model at a time
    
    -- Evaluation metrics (stored as JSONB for flexibility)
    metrics JSONB NOT NULL DEFAULT '{}',
    -- Example metrics structure:
    -- {
    --   "roc_auc": 0.85,
    --   "pr_auc": 0.72,
    --   "brier_score": 0.15,
    --   "precision": 0.68,
    --   "recall": 0.75,
    --   "f1_score": 0.71
    -- }
    
    -- Training metadata
    training_samples INTEGER,
    feature_count INTEGER,
    training_duration_seconds DECIMAL(10, 2),
    hyperparameters JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_metadata_model_id ON model_metadata(model_id);
CREATE INDEX IF NOT EXISTS idx_model_metadata_trained_at ON model_metadata(trained_at);
CREATE INDEX IF NOT EXISTS idx_model_metadata_is_active ON model_metadata(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_model_metadata_model_type ON model_metadata(model_type);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to automatically create geometry from lat/lon
CREATE OR REPLACE FUNCTION update_crimes_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically populate geometry column
DROP TRIGGER IF EXISTS trigger_update_crimes_geom ON crimes;
CREATE TRIGGER trigger_update_crimes_geom
    BEFORE INSERT OR UPDATE ON crimes
    FOR EACH ROW
    EXECUTE FUNCTION update_crimes_geom();

-- Function to ensure only one active model at a time
CREATE OR REPLACE FUNCTION ensure_single_active_model()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = TRUE THEN
        UPDATE model_metadata SET is_active = FALSE WHERE is_active = TRUE AND id != NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to ensure only one active model
DROP TRIGGER IF EXISTS trigger_single_active_model ON model_metadata;
CREATE TRIGGER trigger_single_active_model
    BEFORE INSERT OR UPDATE ON model_metadata
    FOR EACH ROW
    EXECUTE FUNCTION ensure_single_active_model();

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- View: Crime counts by H3 cell and time window
CREATE OR REPLACE VIEW crime_counts_by_h3_time AS
SELECT 
    h3_index,
    DATE_TRUNC('hour', occurred_at) AS time_window,
    COUNT(*) AS crime_count,
    COUNT(DISTINCT crime_type) AS distinct_crime_types
FROM crimes
GROUP BY h3_index, DATE_TRUNC('hour', occurred_at);

-- View: Latest model metrics
CREATE OR REPLACE VIEW latest_model AS
SELECT *
FROM model_metadata
WHERE is_active = TRUE
ORDER BY trained_at DESC
LIMIT 1;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE crimes IS 'Raw crime data from Toronto Open Data, mapped to H3 hexagons';
COMMENT ON TABLE ingestion_metadata IS 'Tracks CSV file ingestion for incremental loading';
COMMENT ON TABLE features IS 'Engineered features for ML training and inference';
COMMENT ON TABLE model_metadata IS 'ML model versions, metrics, and artifact paths';

COMMENT ON COLUMN crimes.h3_index IS 'Uber H3 hexagon index at resolution 9 (approx 300m)';
COMMENT ON COLUMN features.time_window IS '1-hour time window for feature aggregation';
COMMENT ON COLUMN model_metadata.metrics IS 'JSONB object containing evaluation metrics';

