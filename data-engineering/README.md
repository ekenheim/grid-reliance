# Data Engineering

Ingest **ERA5** (ECMWF weather reanalysis) and **ENTSO-E** (load, generation, cross-border flows). ETL writes raw data to Bronze and model-ready Parquet to Silver. Start with ERA5; ENTSO-E follows.

## ERA5 (ECMWF) — Start Here

### 1. Credentials (where to put keys)

**Get your key:** [CDS profile](https://cds.climate.copernicus.eu/profile) — register at [CDS](https://cds.climate.copernicus.eu/), accept ERA5 terms, then create an API key. The key is shown as `UID:APIKEY`.

**Where to put it (pick one):**

- **Option A — `.env` (repo root):** Copy `.env.example` to `.env` and set:
  ```bash
  CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
  CDSAPI_KEY=YOUR_UID:YOUR_API_KEY
  ```
  Do not commit `.env`. The fetch script loads these if set.

- **Option B — `.cdsapirc` file:** Create in repo root or `~`:
  ```ini
  url: https://cds.climate.copernicus.eu/api/v2
  key: <your-uid>:<your-api-key>
  ```

On the cluster, use **ExternalSecrets** (or a sealed secret) to inject the key; do not commit it.

### 2. Fetch ERA5 Nordic (10m wind)

Script: `scripts/fetch_era5_nordic.py`. Requests ERA5 single-levels (10m u/v wind) for a Nordic bounding box, one month per request, and writes NetCDF to disk. Optionally uploads to Bronze (S3-compatible).

```bash
cd data-engineering
pip install -r requirements.txt

# Local: save NetCDF under ./era5_nordic
python scripts/fetch_era5_nordic.py --start 2023-01 --end 2023-12 --output-dir ./era5_nordic

# Upload to Bronze (MinIO or Ceph RGW)
export BRONZE_ENDPOINT=http://localhost:9000
export BRONZE_ACCESS_KEY=minioadmin
export BRONZE_SECRET_KEY=minioadmin
export BRONZE_BUCKET=grid-resilience-bronze
python scripts/fetch_era5_nordic.py --start 2023-01 --end 2023-03 --upload-bronze
```

On the homelab cluster, point `BRONZE_*` at Ceph RGW (from the ObjectBucketClaim secrets). Bucket name: `grid-resilience-bronze`.

### 3. Spark (in-cluster)

Spark jobs read from Bronze and write model-ready Parquet to Silver. Use **SparkApplication** CRDs (kubeflow Spark Operator):

- [processing/spark-era5-ingest.yaml](processing/spark-era5-ingest.yaml) — reference manifest: driver/executor env from Rook Silver (write) and Bronze (read) ObjectBucketClaims; runs [spark_era5_ingest.py](processing/spark_era5_ingest.py) (replace with full ERA5/ENTSO-E ETL as needed).
- [processing/CONTRACT.md](processing/CONTRACT.md) — Silver schema contract: `silver/grid_snapshots.parquet` with columns `timestamp`, `region_id`, `wind_speed_mps` for the inference pipeline.
- `processing/spark-entsoe-ingest.yaml` — ENTSO-E ingest (when ready).

## Storage: Bronze / Silver / Gold

| Tier   | Purpose |
|--------|--------|
| Bronze | Raw ERA5 NetCDF, ENTSO-E XML |
| Silver | Parquet, spatially joined, model-ready |
| Gold   | P10/P50/P90 posterior traces (inference) |

- **Local / dev:** MinIO (e.g. `docker compose`); buckets can be `grid-resilience-bronze`, `grid-resilience-silver`, `grid-resilience-gold` or a single bucket with prefixes.
- **Homelab (Talos / Flux):** Ceph RGW via ObjectBucketClaims: `grid-resilience-bronze`, `grid-resilience-silver`, `grid-resilience-gold`. See [Infrastructure alignment](../docs/infrastructure-alignment.md).

## Container image

The image is based on `apache/spark` and runs as **UID 185** (user `spark`). Platform manifests should set:

```yaml
securityContext:
  runAsUser: 185
  runAsNonRoot: true
  fsGroup: 185
```

## ENTSO-E (later)

- Get a security token from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
- Store via ExternalSecrets when running in-cluster.
- Ingest and spatial join with ERA5 in the Spark ETL step.

## Run (full platform)

See "How to Run" in the root [README](../README.md).
