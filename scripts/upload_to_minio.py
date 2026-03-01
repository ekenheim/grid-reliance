#!/usr/bin/env python3
"""
Upload config and data to MinIO for the Grid Resilience pipeline.

Run from repo root with MinIO env vars set:

  export MINIO_ENDPOINT=minio.example.com
  export MINIO_ACCESS_KEY=...
  export MINIO_SECRET_KEY=...
  python scripts/upload_to_minio.py

  # Upload generator output (Parquet) if present:
  python scripts/upload_to_minio.py --upload-data
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKET = os.environ.get("MINIO_BUCKET", "grid-resilience")
CONFIG_PREFIX = "config/"
RAW_PREFIX = "raw/"


def get_client():
    import boto3
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        print("ERROR: Set MINIO_ENDPOINT (and MINIO_ACCESS_KEY, MINIO_SECRET_KEY)", file=sys.stderr)
        sys.exit(1)
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
    )


def ensure_bucket(client):
    try:
        client.head_bucket(Bucket=BUCKET)
        print(f"Bucket {BUCKET} exists")
    except Exception:
        client.create_bucket(Bucket=BUCKET)
        print(f"Created bucket {BUCKET}")


def upload_zone_config(client):
    """Upload zone centroid config (optional)."""
    config_path = REPO_ROOT / "generator" / "zone_config.json"
    if not config_path.exists():
        print("No zone_config.json found; skipping.")
        return False
    with open(config_path, "rb") as f:
        client.put_object(Bucket=BUCKET, Key=f"{CONFIG_PREFIX}zone_config.json", Body=f)
    print(f"Uploaded {CONFIG_PREFIX}zone_config.json")
    return True


def upload_generator_output(client):
    """Upload generator Parquet output to raw/ prefix."""
    data_dir = REPO_ROOT / "generator" / "output"
    if not data_dir.exists():
        print(f"No {data_dir} found; skipping.")
        return False
    import pandas as pd
    for p in data_dir.glob("*.parquet"):
        key = f"{RAW_PREFIX}{p.name}"
        with open(p, "rb") as f:
            client.put_object(Bucket=BUCKET, Key=key, Body=f)
        print(f"Uploaded {key}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Upload config and data to MinIO")
    ap.add_argument("--upload-data", action="store_true", help="Upload generator output Parquet files")
    args = ap.parse_args()

    client = get_client()
    ensure_bucket(client)

    upload_zone_config(client)
    if args.upload_data:
        upload_generator_output(client)

    print("Done.")


if __name__ == "__main__":
    main()
