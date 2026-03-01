"""
Dagster resource definitions for MinIO (S3-compatible), PostgreSQL, and Redpanda (Phase 2).
"""

import os
import socket
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from dagster import resource, Field, StringSource


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
