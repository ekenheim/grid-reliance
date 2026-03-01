#!/usr/bin/env python3
"""
Full-platform inference driver: read model-ready Parquet from Silver (S3),
train HSGP with Nutpie, compute tail-risk forecasts, write API-compatible
artifacts to Gold (dagster/tail_risk_forecasts.parquet, dagster/hsgp_model.pkl).

Env (Gold = write bucket; Silver = read bucket):
  Gold: BUCKET_NAME, BUCKET_HOST, BUCKET_PORT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        (or MINIO_* equivalents)
  Silver: SILVER_BUCKET_NAME, SILVER_BUCKET_HOST, SILVER_BUCKET_PORT,
          SILVER_AWS_ACCESS_KEY_ID, SILVER_AWS_SECRET_ACCESS_KEY

Silver object key: SILVER_PREFIX/grid_snapshots.parquet (default silver/grid_snapshots.parquet)
  DataFrame must have columns: timestamp, region_id, wind_speed_mps
"""

from __future__ import annotations

import io
import os
import pickle
import sys

import numpy as np
import pandas as pd
import boto3
from botocore.config import Config

# Default zone order (must match pipeline / api.data.DEFAULT_ZONE_IDS)
DEFAULT_ZONE_IDS = ["SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"]
PREFIX = "dagster"
FORECASTS_KEY = f"{PREFIX}/tail_risk_forecasts.parquet"
MODEL_KEY = f"{PREFIX}/hsgp_model.pkl"
TEMPORAL_SCALE_H = 168.0


def _s3_client_from_env(prefix: str = ""):
    """Build (client, bucket) from env. prefix='' for Gold, 'SILVER_' for Silver."""
    if prefix:
        endpoint = os.environ.get(f"{prefix}S3_ENDPOINT", "").strip()
        if not endpoint:
            host = (os.environ.get(f"{prefix}BUCKET_HOST") or "").strip()
            port = (os.environ.get(f"{prefix}BUCKET_PORT") or "").strip()
            if host and port:
                endpoint = f"http://{host}:{port}"
            elif host:
                endpoint = f"http://{host}"
        access = os.environ.get(f"{prefix}AWS_ACCESS_KEY_ID") or os.environ.get(f"{prefix}MINIO_ACCESS_KEY", "")
        secret = os.environ.get(f"{prefix}AWS_SECRET_ACCESS_KEY") or os.environ.get(f"{prefix}MINIO_SECRET_KEY", "")
        bucket = os.environ.get(f"{prefix}BUCKET_NAME") or os.environ.get(f"{prefix}MINIO_BUCKET", "")
    else:
        endpoint = (os.environ.get("MINIO_ENDPOINT") or os.environ.get("S3_ENDPOINT") or "").strip()
        if not endpoint:
            host = (os.environ.get("BUCKET_HOST") or "").strip()
            port = (os.environ.get("BUCKET_PORT") or "").strip()
            if host and port:
                endpoint = f"http://{host}:{port}"
            elif host:
                endpoint = f"http://{host}"
        access = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("MINIO_ACCESS_KEY", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("MINIO_SECRET_KEY", "")
        bucket = os.environ.get("BUCKET_NAME") or os.environ.get("MINIO_BUCKET", "grid-resilience-gold")
    if not endpoint or not access or not secret:
        return None, None
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    region = (
        os.environ.get(f"{prefix}BUCKET_REGION")
        or os.environ.get(f"{prefix}MINIO_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return client, bucket or None


def load_grid_snapshots_from_silver() -> pd.DataFrame:
    """Load grid_snapshots Parquet from Silver bucket."""
    client, bucket = _s3_client_from_env("SILVER_")
    if not client or not bucket:
        raise RuntimeError("Silver not configured: set SILVER_BUCKET_NAME, SILVER_BUCKET_HOST, SILVER_BUCKET_PORT, SILVER_AWS_ACCESS_KEY_ID, SILVER_AWS_SECRET_ACCESS_KEY")
    key = os.environ.get("SILVER_PREFIX", "silver").rstrip("/") + "/grid_snapshots.parquet"
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        return pd.read_parquet(buf)
    except Exception as e:
        raise RuntimeError(f"Failed to load s3://{bucket}/{key}: {e}") from e


def write_forecasts_and_model_to_gold(forecasts_df: pd.DataFrame, model_dict: dict) -> None:
    """Write tail_risk_forecasts.parquet and hsgp_model.pkl to Gold bucket."""
    client, bucket = _s3_client_from_env("")
    if not client or not bucket:
        raise RuntimeError("Gold not configured: set BUCKET_NAME, BUCKET_HOST, BUCKET_PORT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    buf_parquet = io.BytesIO()
    forecasts_df.to_parquet(buf_parquet, index=False)
    buf_parquet.seek(0)
    client.put_object(Bucket=bucket, Key=FORECASTS_KEY, Body=buf_parquet.getvalue())
    buf_pkl = io.BytesIO()
    pickle.dump(model_dict, buf_pkl)
    buf_pkl.seek(0)
    client.put_object(Bucket=bucket, Key=MODEL_KEY, Body=buf_pkl.getvalue())


def main() -> int:
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    from pipeline.model.hsgp_model import (
        build_hsgp_model,
        compute_tail_risk,
        sample_posterior_predictive,
        train_hsgp,
    )

    print("Loading grid_snapshots from Silver...", file=sys.stderr)
    grid_snapshots = load_grid_snapshots_from_silver()
    if grid_snapshots.empty or len(grid_snapshots) < 10:
        print("grid_snapshots empty or too small; exiting.", file=sys.stderr)
        return 1

    # Ensure required column; Spark may use different name
    if "wind_speed_mps" not in grid_snapshots.columns and "wind_speed" in grid_snapshots.columns:
        grid_snapshots = grid_snapshots.rename(columns={"wind_speed": "wind_speed_mps"})

    print("Training HSGP (Nutpie)...", file=sys.stderr)
    result = train_hsgp(
        grid_snapshots,
        m_spatial=5,
        m_temporal=10,
        draws=300,
        tune=200,
        chains=2,
        random_seed=42,
    )
    if result.get("idata") is None:
        print("train_hsgp returned no idata; exiting.", file=sys.stderr)
        return 1

    zone_ids = result.get("metadata", {}).get("zone_ids", DEFAULT_ZONE_IDS)
    if not zone_ids and "region_id" in grid_snapshots.columns:
        zone_ids = sorted(grid_snapshots["region_id"].unique().tolist())
    if not zone_ids:
        zone_ids = list(DEFAULT_ZONE_IDS)

    X_full, y_full, _ = prepare_hsgp_2d_input(grid_snapshots, temporal_scale_h=TEMPORAL_SCALE_H)
    model, gp = build_hsgp_model(X_full, y_full, m_spatial=5, m_temporal=10)
    result_tmp = {"idata": result["idata"], "model": model, "gp": gp}

    grid_snapshots = grid_snapshots.copy()
    grid_snapshots["timestamp"] = pd.to_datetime(grid_snapshots["timestamp"])
    t_max = grid_snapshots["timestamp"].max()
    t_min = grid_snapshots["timestamp"].min()
    base_hours = (t_max - t_min).total_seconds() / 3600.0
    n_zones = len(zone_ids)
    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}
    rows = []
    for horizon_h in (24, 48, 72):
        t_new = t_max + pd.Timedelta(hours=horizon_h)
        temporal_norm = (base_hours + horizon_h) / TEMPORAL_SCALE_H
        spatial_coords_new = np.array(
            [zone_to_idx[z] / max(n_zones - 1, 1) for z in zone_ids], dtype=np.float64
        )
        temporal_coords_new = np.full(n_zones, temporal_norm, dtype=np.float64)
        samples = sample_posterior_predictive(result_tmp, spatial_coords_new, temporal_coords_new, n_samples=500)
        p_shortfall = compute_tail_risk(samples, threshold_mps=3.0)
        for i, zid in enumerate(zone_ids):
            rows.append({
                "timestamp": t_new,
                "region_id": zid,
                "horizon_h": horizon_h,
                "p_shortfall": float(p_shortfall[i]),
            })
    forecasts_df = pd.DataFrame(rows)

    # API expects hsgp_model pickle with spatial_correlation and zone_ids (top-level)
    model_dict = {
        "metadata": result.get("metadata", {}),
        "spatial_correlation": result.get("spatial_correlation"),
        "m_spatial": 5,
        "m_temporal": 10,
        "zone_ids": zone_ids,
    }

    print("Writing to Gold...", file=sys.stderr)
    write_forecasts_and_model_to_gold(forecasts_df, model_dict)
    print(f"Wrote {FORECASTS_KEY} and {MODEL_KEY} to Gold.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
