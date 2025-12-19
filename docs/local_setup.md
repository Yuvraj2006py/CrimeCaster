# Local Development Setup Guide

## Prerequisites

- ✅ Docker Desktop installed and running
- ✅ Python 3.11+ installed
- ✅ Node.js 18+ installed (for frontend)
- ✅ Supabase project created and schema deployed
- ✅ Mapbox access token

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -e ".[dev]"
```

Or if you prefer using a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

### 2. Configure Environment Variables

Make sure your `.env` file has:
- ✅ `DATABASE_URL` - Supabase connection string
- ✅ `MAPBOX_ACCESS_TOKEN` - Your Mapbox token
- ✅ Other Supabase variables

### 3. Run with Docker (Recommended)

```bash
# Build all Docker images
docker-compose build

# Start API and frontend
docker-compose up -d api frontend

# View logs
docker-compose logs -f api
```

Access:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000

### 4. Run Without Docker (Development)

#### Backend (FastAPI)

```bash
# Make sure you're in the project root
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

## Running Data Pipeline Jobs

### Ingest Crime Data

```bash
# With Docker
docker-compose run --rm ingestion

# Without Docker
python -m ingestion.bronze.csv_loader
```

### Train ML Model

```bash
# With Docker
docker-compose run --rm training

# Without Docker
python -m training.trainer
```

## Troubleshooting

### Database Connection Issues

If you're having connection issues:
1. Use Supabase SQL Editor for manual queries
2. Fix `DATABASE_URL` in `.env` with exact connection string from Supabase
3. Test connection: `python scripts/test_connection.py`

### Port Already in Use

If port 8000 or 3000 is already in use:
- Change ports in `docker-compose.yml`
- Or stop the service using the port

### Docker Issues

```bash
# Rebuild containers
docker-compose build --no-cache

# Remove and recreate
docker-compose down
docker-compose up --build
```

## Next Steps

1. ✅ Database schema created
2. ⏳ Set up data ingestion (download Toronto crime data)
3. ⏳ Run feature engineering
4. ⏳ Train initial model
5. ⏳ Start API and frontend
6. ⏳ View 3D map visualization

