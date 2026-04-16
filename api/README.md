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

When S3/MinIO is configured and the **Gold** bucket (or the configured bucket) contains `dagster/tail_risk_forecasts.parquet` and `dagster/hsgp_model.pkl`, the API returns real data (`"status": "ok"`). Otherwise it returns stub data (`"status": "stub"`).

## Container image

The image runs as **UID 1000** (user `api`). Platform manifests should set:

```yaml
securityContext:
  runAsUser: 1000
  runAsNonRoot: true
  fsGroup: 1000
```

## Environment

- **S3-compatible storage (MinIO or Ceph RGW):** Either MinIO-style or Rook-style env.
  - MinIO: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, optional `MINIO_REGION`.
  - Rook ObjectBucketClaim: `BUCKET_HOST`, `BUCKET_PORT`, `BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and optional `BUCKET_REGION`). Point the bucket at **Gold** so the API reads the artifacts written by the inference RayJob or Dagster.
- `PG_CONNECTION_STRING` — reserved for future use (e.g. real forecasts from PostgreSQL).
