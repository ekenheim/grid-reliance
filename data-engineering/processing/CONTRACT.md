# Spark ETL: Bronze → Silver contract

The inference pipeline and API expect **Silver** to contain model-ready Parquet in this form:

- **Key:** `silver/grid_snapshots.parquet` (or configurable prefix + `grid_snapshots.parquet`).
- **Schema:** DataFrame with columns:
  - `timestamp` (datetime)
  - `region_id` (str, e.g. SE1, DK1)
  - `wind_speed_mps` (float) — or `wind_speed` (inference driver accepts both)

Spark (or any ETL) should read raw data from **Bronze** (e.g. ERA5 NetCDF, ENTSO-E), run spatial join / aggregation as needed, and write the above Parquet to **Silver** so that:

1. [inference/run_inference.py](../../inference/run_inference.py) can load it and train HSGP.
2. The RayJob writes `dagster/tail_risk_forecasts.parquet` and `dagster/hsgp_model.pkl` to **Gold**, which the API reads.

## S3/RGW configuration

- **Bronze:** read path `s3a://<BRONZE_BUCKET>/...` (e.g. raw NetCDF or Parquet).
- **Silver:** write path `s3a://<BUCKET_NAME>/silver/grid_snapshots.parquet`.
- Use the same RGW endpoint and credentials for both if your ObjectBucketClaims share the same endpoint (e.g. Ceph RGW).
