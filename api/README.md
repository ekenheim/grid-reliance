# Grid Resilience API

MVP path: FastAPI forecast/correlations. Full platform also exposes a Streamlit Risk Dashboard (see `dashboard/`).

FastAPI service for tail-risk forecasts and spatial correlations.

## Endpoints

- `GET /forecast/{region_id}?horizon_h=24` - P(shortfall > threshold) for 24/48/72h
- `GET /correlations` - Learned spatial correlation matrix

## Local development

**Run API** (from repo root venv, `api` as cwd):

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Server: **http://127.0.0.1:8000**

**Try it**

- Health: http://127.0.0.1:8000/
- Interactive docs: http://127.0.0.1:8000/docs
- Forecast (stub): http://127.0.0.1:8000/forecast/SE1?horizon_h=24
- Correlations (stub): http://127.0.0.1:8000/correlations

Endpoints currently return stub data; wiring to PostgreSQL/MinIO is TODO.

## Environment

- `PG_CONNECTION_STRING` - PostgreSQL (for real forecasts, when implemented)
- `MINIO_*` - MinIO for model artifacts (when implemented)
