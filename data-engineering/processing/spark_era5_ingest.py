"""
ERA5/ENTSO-E ingest: Bronze (Ceph RGW) NetCDF → Silver grid_snapshots.parquet

Spark 4.0.2 job deployed via SparkApplication (Spark Operator) in the datasci namespace.

Bronze layout expected:
  era5/single-levels/era5_nordic_10m_wind_YYYY_MM.nc
  (produced by data-engineering/scripts/fetch_era5_nordic.py --upload-bronze)

Silver output:
  silver/grid_snapshots.parquet
  Schema: timestamp (TimestampType) | region_id (StringType) | wind_speed_mps (DoubleType)

Env vars injected by SparkApplication manifest (from Rook OBC):
  Bronze (read):  BRONZE_BUCKET_HOST, BRONZE_BUCKET_PORT, BRONZE_BUCKET_NAME,
                  BRONZE_AWS_ACCESS_KEY_ID, BRONZE_AWS_SECRET_ACCESS_KEY
  Silver (write): BUCKET_HOST, BUCKET_PORT, BUCKET_NAME,
                  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import io
import os
import sys
from typing import Iterator

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Zone centroids (lat, lon) — must match pipeline/model/feature_engineering.py
ZONE_CENTROIDS: dict[str, tuple[float, float]] = {
    "SE1": (62.0, 14.5),
    "SE2": (62.5, 17.5),
    "SE3": (59.5, 18.0),
    "SE4": (56.0, 14.0),
    "NO1": (59.5, 10.5),
    "NO2": (64.0, 11.5),
    "DK1": (56.5,  9.5),
    "DK2": (55.5, 12.0),
    "FI":  (61.5, 25.0),
}

SILVER_SCHEMA = StructType([
    StructField("timestamp",      TimestampType(), True),
    StructField("region_id",      StringType(),    True),
    StructField("wind_speed_mps", DoubleType(),    True),
])


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client(endpoint: str, access_key: str, secret_key: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _list_nc_keys(client, bucket: str, prefix: str = "era5/single-levels/") -> list[str]:
    """List all .nc object keys under prefix in bucket."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".nc"):
                keys.append(obj["Key"])
    return sorted(keys)


# ---------------------------------------------------------------------------
# NetCDF processing (runs on each Spark executor)
# ---------------------------------------------------------------------------

def _nearest_zone_indices(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    For each ERA5 grid point (lat, lon), return the index of the nearest zone
    centroid using squared Euclidean distance in lat/lon space.

    Returns shape (n_lat * n_lon,) int array.
    """
    centroids = np.array(list(ZONE_CENTROIDS.values()), dtype=np.float32)  # (n_zones, 2)
    lat_g, lon_g = np.meshgrid(lats.astype(np.float32), lons.astype(np.float32), indexing="ij")
    grid = np.stack([lat_g.ravel(), lon_g.ravel()], axis=1)             # (N, 2)
    dists = np.sum((grid[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (N, n_zones)
    return np.argmin(dists, axis=1)                                      # (N,)


def _process_nc_bytes(nc_bytes: bytes) -> pd.DataFrame:
    """
    Convert raw ERA5 NetCDF bytes to a DataFrame with columns
    [timestamp, region_id, wind_speed_mps].

    Handles both ERA5-CDS-API-v1 ("time") and v2 ("valid_time") dimension names.
    Computes wind speed magnitude from u10 and v10 components and averages
    all ERA5 grid points assigned to each Nordic zone.
    """
    import xarray as xr

    buf = io.BytesIO(nc_bytes)
    with xr.open_dataset(buf, engine="netcdf4") as ds:
        # ERA5 CDS API ≥2024 names the time dimension "valid_time"
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        times = pd.to_datetime(ds[time_dim].values)

        lats = ds["latitude"].values.astype(np.float32)
        lons = ds["longitude"].values.astype(np.float32)

        u10 = ds["u10"].values.astype(np.float32)  # (time, lat, lon)
        v10 = ds["v10"].values.astype(np.float32)  # (time, lat, lon)

    # Wind speed magnitude at every grid point
    ws = np.sqrt(u10 ** 2 + v10 ** 2)  # (time, lat, lon)

    zone_ids = list(ZONE_CENTROIDS.keys())
    n_zones = len(zone_ids)
    zone_assignment = _nearest_zone_indices(lats, lons)  # (n_lat * n_lon,)

    rows: list[tuple] = []
    for t_idx, ts in enumerate(times):
        ws_flat = ws[t_idx].ravel()
        ts_py = ts.to_pydatetime()
        for z_idx in range(n_zones):
            mask = zone_assignment == z_idx
            if mask.any():
                rows.append((ts_py, zone_ids[z_idx], float(np.mean(ws_flat[mask]))))

    return pd.DataFrame(rows, columns=["timestamp", "region_id", "wind_speed_mps"])


def _process_nc_key(args: tuple) -> list[tuple]:
    """
    Spark RDD map function executed on each executor.

    args: (s3_key, bronze_endpoint, bronze_access, bronze_secret, bronze_bucket)
    Returns list of (timestamp, region_id, wind_speed_mps) Row tuples.
    """
    s3_key, endpoint, access_key, secret_key, bucket = args
    client = _s3_client(endpoint, access_key, secret_key)
    try:
        nc_bytes = client.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
        df = _process_nc_bytes(nc_bytes)
        return list(df.itertuples(index=False, name=None))
    except Exception as exc:
        print(f"[WARN] Skipping {s3_key}: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Silver write helper (boto3 + pyarrow — avoids hadoop-aws dependency)
# ---------------------------------------------------------------------------

def _write_parquet_to_silver(pdf: pd.DataFrame, endpoint: str, access_key: str,
                              secret_key: str, bucket: str, key: str) -> None:
    """Write a pandas DataFrame as Parquet directly to Silver via boto3."""
    buf = io.BytesIO()
    pdf.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    client = _s3_client(endpoint, access_key, secret_key)
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Bronze credentials (read)
    bronze_host   = (os.environ.get("BRONZE_BUCKET_HOST") or "").strip()
    bronze_port   = (os.environ.get("BRONZE_BUCKET_PORT") or "80").strip()
    bronze_access = (os.environ.get("BRONZE_AWS_ACCESS_KEY_ID") or "").strip()
    bronze_secret = (os.environ.get("BRONZE_AWS_SECRET_ACCESS_KEY") or "").strip()
    bronze_bucket = (os.environ.get("BRONZE_BUCKET_NAME") or "grid-resilience-bronze").strip()

    # Silver credentials (write)
    silver_host   = (os.environ.get("SILVER_BUCKET_HOST") or "").strip()
    silver_port   = (os.environ.get("SILVER_BUCKET_PORT") or "80").strip()
    silver_access = (os.environ.get("SILVER_AWS_ACCESS_KEY_ID") or "").strip()
    silver_secret = (os.environ.get("SILVER_AWS_SECRET_ACCESS_KEY") or "").strip()
    silver_bucket = (os.environ.get("SILVER_BUCKET_NAME") or "grid-resilience-silver").strip()

    if not bronze_host:
        print("BRONZE_BUCKET_HOST not set — cannot read from Bronze.", file=sys.stderr)
        return 1
    if not silver_host:
        print("SILVER_BUCKET_HOST not set — cannot write to Silver.", file=sys.stderr)
        return 1

    bronze_endpoint = f"http://{bronze_host}:{bronze_port}"
    silver_endpoint = f"http://{silver_host}:{silver_port}"
    silver_key = "silver/grid_snapshots.parquet"

    # List ERA5 NetCDF files
    bronze_client = _s3_client(bronze_endpoint, bronze_access, bronze_secret)
    nc_keys = _list_nc_keys(bronze_client, bronze_bucket, prefix="era5/single-levels/")
    if not nc_keys:
        print(
            "No ERA5 NetCDF files found at era5/single-levels/ in Bronze. "
            "Run fetch_era5_nordic.py --upload-bronze first.",
            file=sys.stderr,
        )
        return 1
    print(f"Found {len(nc_keys)} ERA5 NetCDF file(s) in Bronze.", file=sys.stderr)

    # Spark session — no S3A needed (reads via boto3 in UDFs, writes via boto3 at end)
    spark = (
        SparkSession.builder
        .appName("grid-resilience-era5-ingest")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Distribute file processing: one Spark task per .nc file
    n_slices = min(len(nc_keys), 32)
    args_rdd = spark.sparkContext.parallelize(
        [(k, bronze_endpoint, bronze_access, bronze_secret, bronze_bucket) for k in nc_keys],
        numSlices=n_slices,
    )
    rows_rdd = args_rdd.flatMap(_process_nc_key)
    df_spark = spark.createDataFrame(rows_rdd, schema=SILVER_SCHEMA)

    # Deduplicate (same (timestamp, region_id) could appear across overlapping files)
    df_spark = (
        df_spark
        .groupBy("timestamp", "region_id")
        .agg(F.mean("wind_speed_mps").alias("wind_speed_mps"))
        .orderBy("timestamp", "region_id")
    )

    row_count = df_spark.count()
    print(f"Processed {row_count:,} rows from {len(nc_keys)} file(s).", file=sys.stderr)
    if row_count == 0:
        print("Zero rows — check Bronze NetCDF variable names (u10/v10/latitude/longitude).",
              file=sys.stderr)
        return 1

    # Collect to pandas and write via boto3 (avoids hadoop-aws/s3a dependency in image)
    pdf = df_spark.toPandas()
    _write_parquet_to_silver(pdf, silver_endpoint, silver_access, silver_secret,
                             silver_bucket, silver_key)
    print(f"Wrote {row_count:,} rows to s3://{silver_bucket}/{silver_key}", file=sys.stderr)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
