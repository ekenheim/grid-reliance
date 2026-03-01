"""
Load forecast and correlation data from MinIO (same layout as Dagster pipeline writes).
"""

import io
import os
import pickle

import pandas as pd
import boto3
from botocore.config import Config

# Dagster IO manager prefix and keys
PREFIX = "dagster"
FORECASTS_KEY = f"{PREFIX}/tail_risk_forecasts.parquet"
MODEL_KEY = f"{PREFIX}/hsgp_model.pkl"

# Default zone order for correlation matrix (must match pipeline)
DEFAULT_ZONE_IDS = ["SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"]


def _get_s3_client():
    """Return (s3_client, bucket) or (None, None) if MinIO not configured."""
    endpoint = (os.environ.get("MINIO_ENDPOINT") or "").strip()
    if not endpoint:
        return None, None
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", ""),
        region_name=os.environ.get("MINIO_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    bucket = os.environ.get("MINIO_BUCKET", "grid-resilience")
    return client, bucket


def get_forecast_for_region(region_id: str, horizon_h: int) -> float | None:
    """
    Read tail_risk_forecasts from MinIO and return p_shortfall for the given region and horizon.
    Returns None if object missing, empty, or no matching row.
    """
    client, bucket = _get_s3_client()
    if not client or not bucket:
        return None
    try:
        response = client.get_object(Bucket=bucket, Key=FORECASTS_KEY)
        buf = io.BytesIO(response["Body"].read())
        df = pd.read_parquet(buf)
    except Exception:
        return None
    if df.empty or "region_id" not in df.columns or "p_shortfall" not in df.columns:
        return None
    if "horizon_h" in df.columns:
        row = df[(df["region_id"] == region_id) & (df["horizon_h"] == horizon_h)]
    else:
        row = df[df["region_id"] == region_id]
    if row.empty:
        return None
    return float(row["p_shortfall"].iloc[0])


def get_spatial_correlation() -> tuple[list[str], list[list[float]]] | None:
    """
    Read hsgp_model pickle from MinIO and return (zone_ids, correlation_matrix).
    Returns None if object missing or spatial_correlation not present.
    """
    client, bucket = _get_s3_client()
    if not client or not bucket:
        return None
    try:
        response = client.get_object(Bucket=bucket, Key=MODEL_KEY)
        obj = pickle.loads(response["Body"].read())
    except Exception:
        return None
    if not isinstance(obj, dict) or "spatial_correlation" not in obj:
        return None
    corr = obj["spatial_correlation"]
    if corr is None:
        return None
    # Support numpy array or list of lists
    if hasattr(corr, "tolist"):
        matrix = corr.tolist()
    elif isinstance(corr, (list, tuple)):
        matrix = [list(row) for row in corr]
    else:
        return None
    zone_ids = obj.get("zone_ids", DEFAULT_ZONE_IDS)
    if hasattr(zone_ids, "tolist"):
        zone_ids = zone_ids.tolist()
    return list(zone_ids), matrix
