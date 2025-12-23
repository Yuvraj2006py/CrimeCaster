# Crime Caster - Toronto

An end-to-end batch data pipeline and ML forecasting system that predicts aggregated crime risk probability for Toronto using historical crime data.

## 🎯 Overview

Crime Caster ingests public Toronto crime data, engineers spatiotemporal features, trains interpretable ML models, and serves predictions via an API with an interactive 3D map visualization.

**Key Features:**
- Batch data pipeline (Bronze → Silver → Gold layers)
- **Local Parquet storage** for unlimited ML training data
- Spatial indexing using Uber H3 hexagons
- Temporal feature engineering
- ML model training and versioning
- FastAPI backend for inference
- 3D interactive map (Mapbox GL JS + Deck.gl)
- Dockerized deployment

## ⚠️ Important Disclaimers

- **Predictions represent historical pattern-based risk, not certainty**
- **No individual predictions or personal identifiers**
- **Aggregated spatial units only**
- This system is for informational purposes and does not provide policing or enforcement recommendations

## 🏗️ Architecture

```
Toronto Open Data CSV → Bronze Layer → Silver Layer → Gold Layer
                                                          ↓
                    ┌─────────────────────────────────────┴─────────────────────┐
                    ↓                                                           ↓
            Parquet Files (local)                                    PostgreSQL (Neon, optional)
                    ↓                                                           ↓
            ML Training (unlimited)                                API Serving (limited)
                    ↓                                                           ↓
            Model Artifacts → FastAPI → React 3D Map
```

**Storage Modes:**
- **Hybrid (Default)**: Parquet files (unlimited) + Database (for API)
- **Local-Only**: Parquet files only (perfect for ML training)
- See [Local Storage Guide](docs/local_storage_guide.md) for details

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Neon account (free tier available) or any PostgreSQL with PostGIS
- Mapbox account (free tier available)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CrimeCaster
```

### 2. Set Up Database

**Recommended: Neon (Free, No Pausing)**

1. Create a project at [neon.tech](https://neon.tech) (free)
2. Enable PostGIS extension:
   - Go to SQL Editor in Neon dashboard
   - Run: `CREATE EXTENSION IF NOT EXISTS postgis;`
3. Get connection string from Neon dashboard → Connection Details

**Alternative Providers:**
- Railway, Render, Local Docker, or any PostgreSQL with PostGIS
- See [Database Setup Guide](docs/database_setup.md) for details

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `DATABASE_URL` - PostgreSQL connection string (from Neon, Railway, etc.)
- `MAPBOX_ACCESS_TOKEN` - Mapbox access token (get from [mapbox.com](https://account.mapbox.com/access-tokens/))

**Quick Neon Setup:** See [README_NEON_SETUP.md](README_NEON_SETUP.md) for 5-minute setup guide.

### 4. Build and Run with Docker

```bash
# Build all services
docker-compose build

# Start API and frontend
docker-compose up -d api frontend

# Run database migrations (first time setup)
docker-compose run --rm api python -m sql.migrations.run_migrations

# Run ingestion (download and process crime data)
docker-compose run --rm ingestion

# Train initial model
docker-compose run --rm training
```

### 5. Access the Application

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000

## 📁 Project Structure

```
crime-caster/
├── ingestion/          # Data ingestion pipeline
│   ├── bronze/         # Raw CSV loading
│   └── silver/         # Initial cleaning
├── validation/         # Data validation
├── transformations/    # Data transformations
│   ├── silver/        # Cleaning & normalization
│   └── gold/          # H3 mapping & DB loading
├── feature_engineering/ # Feature creation
├── training/          # ML model training
│   └── models/        # Model implementations
├── inference/         # Real-time prediction
├── api/              # FastAPI backend
│   └── routes/       # API endpoints
├── frontend/         # Next.js frontend
│   ├── components/   # React components
│   └── lib/          # Utilities
├── notebooks/        # Jupyter notebooks
├── sql/              # Database schemas & migrations
├── docs/             # Documentation
└── monitoring/       # Monitoring & alerts
```

## 🐳 Docker Services

- **api**: FastAPI backend (port 8000)
- **frontend**: Next.js frontend (port 3000)
- **ingestion**: Batch data ingestion (run on-demand)
- **training**: ML model training (run on-demand)

### Useful Docker Commands

```bash
# View logs
docker-compose logs -f api

# Run ingestion job
docker-compose run --rm ingestion

# Run training job
docker-compose run --rm training

# Stop all services
docker-compose down

# Rebuild after dependency changes
docker-compose build --no-cache api
```

## 🔄 Data Pipeline

### Weekly Ingestion

```bash
docker-compose run --rm ingestion
```

This will:
1. Download latest crime CSV from Toronto Open Data
2. Validate and clean data
3. Map to H3 hexagons
4. Load into Supabase incrementally

### Monthly Training

```bash
docker-compose run --rm training
```

This will:
1. Generate features for all H3 cells
2. Train ML models
3. Evaluate and version models
4. Store model artifacts

## 🧪 Development

### Local Development (without Docker)

```bash
# Install Python dependencies
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend && npm install

# Run API locally
uvicorn api.main:app --reload

# Run frontend locally
cd frontend && npm run dev
```

### Running Tests

```bash
pytest
```

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /risk` - Get risk score for location + datetime
- `GET /heatmap` - Get risk heatmap for bounding box
- `GET /model-info` - Get current model version and metrics

See http://localhost:8000/docs for interactive API documentation.

## 🗺️ 3D Map Features

- Interactive 3D navigation (pan, zoom, rotate, tilt)
- H3 hexagon risk visualization as 3D extruded polygons
- Click to navigate to specific locations
- Time slider for temporal exploration
- Crime type filters
- Risk explanation tooltips

## 📚 Documentation

- [Architecture](docs/architecture.md)
- [Data Pipeline](docs/data_pipeline.md)
- [API Documentation](docs/api_documentation.md)
- [Docker Setup](docs/docker.md)
- [Supabase Setup](docs/supabase_setup.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

This is a portfolio project. Contributions and feedback are welcome!

## 📝 License

MIT License

## 🙏 Acknowledgments

- Toronto Police Service Open Data Portal
- Supabase for managed PostgreSQL
- Mapbox for mapping services
- Uber H3 for spatial indexing

