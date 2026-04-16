# Inference

KubeRay RayJob for HSGP inference using Nutpie. Reads model-ready Parquet from **Silver** (Rook-Ceph or MinIO), runs NUTS sampling, and writes API-compatible artifacts to **Gold** so the FastAPI service serves real forecasts and correlations.

## Layout

```
inference/
├── bayesian-grid-model.yaml   # RayJob manifest (use existing Ray cluster in datasci)
├── run_inference.py            # Driver: Silver -> train HSGP -> write Gold (dagster/*)
├── __init__.py
└── README.md
```

## Inputs / Outputs

- **Inputs:** Silver bucket, key `silver/grid_snapshots.parquet` (or `SILVER_PREFIX/grid_snapshots.parquet`). DataFrame must have columns `timestamp`, `region_id`, `wind_speed_mps` (or `wind_speed`).
- **Outputs:** Gold bucket, keys `dagster/tail_risk_forecasts.parquet` and `dagster/hsgp_model.pkl` (same layout the API and pipeline expect).

## Environment (RayJob / submitter pod)

**Gold (write):** From Rook ObjectBucketClaim for `grid-resilience-gold` via `envFrom`:

- `BUCKET_NAME`, `BUCKET_HOST`, `BUCKET_PORT`, `BUCKET_REGION`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

**Silver (read):** From Rook ObjectBucketClaim for `grid-resilience-silver`, mapped with `SILVER_` prefix in the manifest:

- `SILVER_BUCKET_NAME`, `SILVER_BUCKET_HOST`, `SILVER_BUCKET_PORT`
- `SILVER_AWS_ACCESS_KEY_ID`, `SILVER_AWS_SECRET_ACCESS_KEY`

Optional: `SILVER_PREFIX` (default `silver`) for the object key prefix.

## Container image

The image runs as **UID 1000** (user `inference`). Platform manifests should set:

```yaml
securityContext:
  runAsUser: 1000
  runAsNonRoot: true
  fsGroup: 1000
```

## Run

- **In-cluster:** Apply the RayJob (after fixing image and secret/configMap names in [bayesian-grid-model.yaml](bayesian-grid-model.yaml)):

  ```bash
  kubectl apply -f inference/bayesian-grid-model.yaml
  ```

- **Image:** Build an image that includes this repo and pipeline deps (PyMC, Nutpie, boto3, pandas). Use repo root as working dir so `python -m inference.run_inference` runs. See [docs/infrastructure-alignment.md](../docs/infrastructure-alignment.md) for optional custom Ray image via Harbor.

- Root [README](../README.md) "How to Run" describes the full platform flow (Spark -> Silver, RayJob -> Gold, API reads Gold).
