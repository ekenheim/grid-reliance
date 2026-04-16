# Grid Resilience Pipeline

MVP path: Dagster + PyMC HSGP on synthetic data. Full platform uses KubeRay + Nutpie for distributed inference (see `inference/`).

HSGP model training and Dagster orchestration for Pan-European Grid Resilience.

## Components

- **dagster/**: Asset definitions (raw_weather_obs → grid_snapshots → hsgp_model → tail_risk_forecasts → risk_alerts)
- **model/**: HSGP model (PyMC/NumPyro), feature engineering, diagnostics, MLflow
- **streaming/**: Bytewax dataflows for Phase 2 (spatial windowing, anomaly detection)

## Local development

**Start PostgreSQL (from repo root):**

```bash
docker compose up -d postgres
```

**Run Dagster:** Must run from repo root (not from `pipeline/`) so the code server does not shadow the `dagster` package. Use the project venv so `dagster` is on PATH:

```bash
# From repo root (e.g. cd to proj-grid-resilience)
# Activate venv first (Windows PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows CMD:  .venv\Scripts\activate.bat
# macOS/Linux:  source .venv/bin/activate

pip install -r pipeline/requirements.txt
dagster dev -m pipeline.dagster.repository
```

If you don't activate the venv, run via the venv Python:
`.\.venv\Scripts\python.exe -m dagster dev -m pipeline.dagster.repository` (Windows).

Or: `dg dev -m pipeline.dagster.repository` (if the Dagster CLI is installed).

## Container image

The image runs as **UID 1000** (user `pipeline`). Platform manifests should set:

```yaml
securityContext:
  runAsUser: 1000
  runAsNonRoot: true
  fsGroup: 1000
```

## Environment variables

- `PG_CONNECTION_STRING` (PostgreSQL; default for local: `postgresql://grid_resilience:grid_resilience@localhost:5432/grid_resilience`)
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`
- `REDPANDA_BROKERS` (Phase 2)
- `MLFLOW_TRACKING_URI`

## Phase 1 (MVP)

- Batch load from PostgreSQL
- HSGP model in PyMC or NumPyro
- 6-hourly retrain schedule

## Phase 2

- Bytewax stream processing
- Redpanda integration
- Risk alert publishing
