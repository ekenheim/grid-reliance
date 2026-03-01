"""
Dagster resource definitions for Rook Ceph S3 (Silver/Gold), MinIO (local dev fallback),
PostgreSQL (local dev fallback), and Redpanda (Phase 2).
"""

import os
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from dagster import resource, Field, StringSource


def _rook_s3_client(host: str, port: str, access_key: str, secret_key: str):
    endpoint = f"http://{host}:{port}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=15,       # seconds to establish TCP connection
            read_timeout=120,         # seconds to wait for a response once connected
            retries={
                "max_attempts": 5,
                "mode": "adaptive",   # back-off on throttling + transient errors
            },
        ),
    )


@resource
def silver_resource(context):
    """Rook Ceph Silver bucket (read-only: grid snapshots from Spark ERA5/ENTSO-E ingest).

    Returns None when Silver env vars are not set so that raw_weather_obs can
    fall back to the Postgres synthetic dataset without crashing on startup.
    """
    host = os.environ.get("SILVER_BUCKET_HOST", "").strip()
    port = os.environ.get("SILVER_BUCKET_PORT", "").strip()
    access_key = os.environ.get("SILVER_AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("SILVER_AWS_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("SILVER_BUCKET_NAME", "").strip()
    if not all([host, port, access_key, secret_key, bucket]):
        context.log.info(
            "Silver bucket not configured (SILVER_BUCKET_HOST/PORT/CREDS absent) — "
            "raw_weather_obs will fall back to Postgres."
        )
        return None
    client = _rook_s3_client(host=host, port=port, access_key=access_key, secret_key=secret_key)
    return {"client": client, "bucket": bucket}


@resource
def bronze_resource(context):
    """Rook Ceph Bronze bucket (write: ERA5 NetCDF and ENTSO-E CSV files from fetch assets).

    Returns None when Bronze env vars are absent so that fetch assets can skip
    upload gracefully during local development.
    """
    host = os.environ.get("BRONZE_BUCKET_HOST", "").strip()
    port = os.environ.get("BRONZE_BUCKET_PORT", "").strip()
    access_key = os.environ.get("BRONZE_AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("BRONZE_AWS_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("BRONZE_BUCKET_NAME", "").strip()
    if not all([host, port, access_key, secret_key, bucket]):
        context.log.info(
            "Bronze bucket not configured (BRONZE_BUCKET_HOST/PORT/CREDS absent) — "
            "fetch assets will write locally only."
        )
        return None
    client = _rook_s3_client(host=host, port=port, access_key=access_key, secret_key=secret_key)
    return {"client": client, "bucket": bucket}


@resource
def gold_resource(context):
    """Rook Ceph Gold bucket (read-write: Dagster pipeline outputs)."""
    client = _rook_s3_client(
        host=os.environ["GOLD_BUCKET_HOST"],
        port=os.environ["GOLD_BUCKET_PORT"],
        access_key=os.environ["GOLD_AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["GOLD_AWS_SECRET_ACCESS_KEY"],
    )
    return {"client": client, "bucket": os.environ["GOLD_BUCKET_NAME"]}


@resource(
    config_schema={
        "endpoint_url": Field(StringSource, default_value=os.environ.get("MINIO_ENDPOINT", "")),
        "access_key": Field(StringSource, default_value=os.environ.get("MINIO_ACCESS_KEY", "")),
        "secret_key": Field(StringSource, default_value=os.environ.get("MINIO_SECRET_KEY", "")),
        "bucket": Field(str, default_value=os.environ.get("MINIO_BUCKET", "grid-resilience")),
    }
)
def minio_resource(context):
    """S3-compatible MinIO client resource."""
    cfg = context.resource_config
    endpoint = (cfg["endpoint_url"] or "").strip()
    if not endpoint:
        raise ValueError(
            "MINIO endpoint is not configured. Set MINIO_ENDPOINT to a reachable host:port."
        )
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        raise ValueError(f"Invalid MINIO_ENDPOINT '{endpoint}'.")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name=os.environ.get("MINIO_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    bucket = cfg["bucket"] or "grid-resilience"
    return {
        "client": client,
        "bucket": bucket,
        "endpoint_url": endpoint,
        "access_key": cfg["access_key"],
        "secret_key": cfg["secret_key"],
    }


@resource(
    config_schema={
        "connection_string": Field(
            StringSource,
            default_value=os.environ.get(
                "PG_CONNECTION_STRING",
                "postgresql://grid_resilience:grid_resilience@postgres:5432/grid_resilience",
            ),
        ),
    }
)
def postgres_resource(context):
    """PostgreSQL connection resource (returns a connection string)."""
    return context.resource_config["connection_string"]


@resource(
    config_schema={
        "bootstrap_servers": Field(
            StringSource,
            default_value=os.environ.get("REDPANDA_BROKERS", "redpanda:9092"),
        ),
    }
)
def redpanda_resource(context):
    """Redpanda/Kafka resource for Phase 2 risk alert publishing."""
    return {"bootstrap_servers": context.resource_config["bootstrap_servers"]}
