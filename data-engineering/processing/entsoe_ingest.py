"""
ENTSO-E ingest: Bronze CSV → Silver entsoe_snapshots.parquet

Spark 4.0.2 job deployed via SparkApplication (Spark Operator) in the datasci namespace.

Bronze layout expected (uploaded by data-engineering/scripts/fetch_entsoe.py):
  entsoe/generation/actual_generation_YYYY_MM.csv
  entsoe/load/actual_load_YYYY_MM.csv

Both CSV files share the schema produced by fetch_entsoe.py:
  timestamp (ISO-8601), region_id (bidding zone code), variable (str), value_mwh (float)

Silver output:
  silver/entsoe_snapshots.parquet
  Schema: timestamp (TimestampType) | region_id (StringType)
         | wind_onshore_mwh (DoubleType) | wind_offshore_mwh (DoubleType)
         | total_load_mwh (DoubleType)

Env vars injected by SparkApplication manifest (from Rook OBC):
  Bronze (read):  BRONZE_BUCKET_HOST, BRONZE_BUCKET_PORT, BRONZE_BUCKET_NAME,
                  BRONZE_AWS_ACCESS_KEY_ID, BRONZE_AWS_SECRET_ACCESS_KEY
  Silver (write): SILVER_BUCKET_HOST, SILVER_BUCKET_PORT, SILVER_BUCKET_NAME,
                  SILVER_AWS_ACCESS_KEY_ID, SILVER_AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import io
import os
import sys

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

# ENTSO-E zone/area codes → our internal region_id.
# Generation (A75) is published at CONTROL-AREA level for SE and NO, so
# fetch_entsoe.py writes region_id "SE" / "NO" for those rows.
# Load (A65) is published at bidding-zone level (SE1-SE4, NO1-NO2 …).
ZONE_MAP: dict[str, str] = {
    # Bidding-zone short codes (load rows)
    "SE1": "SE1", "SE2": "SE2", "SE3": "SE3", "SE4": "SE4",
    "NO1": "NO1", "NO2": "NO2",
    "DK1": "DK1", "DK2": "DK2",
    "FI":  "FI",
    # Control-area short codes (generation rows — SE and NO at system level)
    "SE":  "SE",
    "NO":  "NO",
    # Full EIC codes as fallback (API sometimes embeds them in the response)
    "10Y1001A1001A44P": "SE1",
    "10Y1001A1001A45N": "SE2",
    "10Y1001A1001A46L": "SE3",
    "10Y1001A1001A47J": "SE4",
    "10YNO-1--------2": "NO1",
    "10YNO-2--------T": "NO2",
    "10YDK-1--------W": "DK1",
    "10YDK-2--------M": "DK2",
    "10YFI-1--------U": "FI",
    "10YSE-1--------K": "SE",   # Sweden system (generation control area)
    "10YNO-0--------C": "NO",   # Norway system (generation control area)
}

# ENTSO-E production type codes written as the `variable` column by fetch_entsoe.py
WIND_ONSHORE_CODE  = "B19"   # Wind Onshore
WIND_OFFSHORE_CODE = "B18"   # Wind Offshore
LOAD_VARIABLE      = "load_mwh"   # written by fetch_entsoe._parse_load

SILVER_SCHEMA = StructType([
    StructField("timestamp",          TimestampType(), True),
    StructField("region_id",          StringType(),    True),
    StructField("wind_onshore_mwh",   DoubleType(),    True),
    StructField("wind_offshore_mwh",  DoubleType(),    True),
    StructField("total_load_mwh",     DoubleType(),    True),
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


def _list_csv_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"])
    return sorted(keys)


def _read_csv_from_s3(client, bucket: str, key: str) -> pd.DataFrame:
    resp = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


def _write_parquet_to_silver(pdf: pd.DataFrame, endpoint: str, access_key: str,
                              secret_key: str, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    pdf.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    client = _s3_client(endpoint, access_key, secret_key)
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------------------------------------------------------------------------
# Processing helpers (run in Spark executor via RDD map)
# ---------------------------------------------------------------------------

def _process_generation_key(args: tuple) -> list[tuple]:
    """
    Read one ENTSO-E generation CSV from Bronze, extract wind onshore/offshore,
    return list of (timestamp, region_id, wind_onshore_mwh, wind_offshore_mwh, None) tuples.
    """
    key, endpoint, access_key, secret_key, bucket = args
    client = _s3_client(endpoint, access_key, secret_key)
    try:
        df = _read_csv_from_s3(client, bucket, key)
        # Expected columns: timestamp, region_id (or zone_code), variable, value_mwh
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region_id"] = df["region_id"].map(lambda z: ZONE_MAP.get(z, z))
        df = df[df["region_id"].isin(ZONE_MAP.values())]

        onshore  = df[df["variable"] == WIND_ONSHORE_CODE ].copy()
        offshore = df[df["variable"] == WIND_OFFSHORE_CODE].copy()

        onshore  = onshore .groupby(["timestamp", "region_id"])["value_mwh"].sum().reset_index()
        offshore = offshore.groupby(["timestamp", "region_id"])["value_mwh"].sum().reset_index()

        merged = onshore.merge(offshore, on=["timestamp", "region_id"],
                               how="outer", suffixes=("_on", "_off"))
        merged.columns = ["timestamp", "region_id", "wind_onshore_mwh", "wind_offshore_mwh"]
        merged = merged.fillna(0.0)

        return [
            (row.timestamp.to_pydatetime(), row.region_id,
             float(row.wind_onshore_mwh), float(row.wind_offshore_mwh), None)
            for row in merged.itertuples(index=False)
        ]
    except Exception as exc:
        print(f"[WARN] Skipping generation file {key}: {exc}", file=sys.stderr)
        return []


def _process_load_key(args: tuple) -> list[tuple]:
    """
    Read one ENTSO-E load CSV from Bronze, return list of
    (timestamp, region_id, None, None, total_load_mwh) tuples.
    """
    key, endpoint, access_key, secret_key, bucket = args
    client = _s3_client(endpoint, access_key, secret_key)
    try:
        df = _read_csv_from_s3(client, bucket, key)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region_id"] = df["region_id"].map(lambda z: ZONE_MAP.get(z, z))
        df = df[df["region_id"].isin(ZONE_MAP.values())]
        df = df[df["variable"] == LOAD_VARIABLE]
        agg = df.groupby(["timestamp", "region_id"])["value_mwh"].sum().reset_index()
        return [
            (row.timestamp.to_pydatetime(), row.region_id, None, None, float(row.value_mwh))
            for row in agg.itertuples(index=False)
        ]
    except Exception as exc:
        print(f"[WARN] Skipping load file {key}: {exc}", file=sys.stderr)
        return []


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
    silver_key = "silver/entsoe_snapshots.parquet"

    bronze_client = _s3_client(bronze_endpoint, bronze_access, bronze_secret)
    gen_keys  = _list_csv_keys(bronze_client, bronze_bucket, prefix="entsoe/generation/")
    load_keys = _list_csv_keys(bronze_client, bronze_bucket, prefix="entsoe/load/")

    if not gen_keys and not load_keys:
        print(
            "No ENTSO-E CSV files found at entsoe/generation/ or entsoe/load/ in Bronze. "
            "Run fetch_entsoe.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(gen_keys)} generation + {len(load_keys)} load CSV file(s).", file=sys.stderr)

    # Freshness check: skip if Silver already reflects all Bronze CSV files.
    silver_client = _s3_client(silver_endpoint, silver_access, silver_secret)
    try:
        silver_head = silver_client.head_object(Bucket=silver_bucket, Key=silver_key)
        silver_mtime = silver_head["LastModified"]
        all_bronze_keys = gen_keys + load_keys
        newest_bronze_mtime = max(
            bronze_client.head_object(Bucket=bronze_bucket, Key=k)["LastModified"]
            for k in all_bronze_keys
        )
        if silver_mtime >= newest_bronze_mtime:
            print(
                f"Silver entsoe_snapshots.parquet is up-to-date "
                f"(Silver: {silver_mtime.isoformat()}, newest Bronze: {newest_bronze_mtime.isoformat()}). "
                "Nothing to do.",
                file=sys.stderr,
            )
            return 0
        print(
            f"New Bronze data detected (newest Bronze: {newest_bronze_mtime.isoformat()}, "
            f"Silver: {silver_mtime.isoformat()}). Re-ingesting.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"Freshness check skipped ({exc}); proceeding with full ingest.", file=sys.stderr)

    spark = (
        SparkSession.builder
        .appName("grid-resilience-entsoe-ingest")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    _ROW_SCHEMA = SILVER_SCHEMA  # (timestamp, region_id, wind_on, wind_off, load)

    all_rows_rdd = spark.sparkContext.emptyRDD()

    if gen_keys:
        gen_rdd = spark.sparkContext.parallelize(
            [(k, bronze_endpoint, bronze_access, bronze_secret, bronze_bucket) for k in gen_keys],
            numSlices=min(len(gen_keys), 16),
        ).flatMap(_process_generation_key)
        all_rows_rdd = all_rows_rdd.union(gen_rdd)

    if load_keys:
        load_rdd = spark.sparkContext.parallelize(
            [(k, bronze_endpoint, bronze_access, bronze_secret, bronze_bucket) for k in load_keys],
            numSlices=min(len(load_keys), 16),
        ).flatMap(_process_load_key)
        all_rows_rdd = all_rows_rdd.union(load_rdd)

    df_spark = spark.createDataFrame(all_rows_rdd, schema=SILVER_SCHEMA)

    # Pivot: union of generation rows and load rows — aggregate to one row per (ts, zone)
    df_out = (
        df_spark
        .groupBy("timestamp", "region_id")
        .agg(
            F.sum("wind_onshore_mwh").alias("wind_onshore_mwh"),
            F.sum("wind_offshore_mwh").alias("wind_offshore_mwh"),
            F.sum("total_load_mwh").alias("total_load_mwh"),
        )
        .orderBy("timestamp", "region_id")
    )

    row_count = df_out.count()
    print(f"Processed {row_count:,} rows.", file=sys.stderr)
    if row_count == 0:
        print("Zero rows — check Bronze CSV schema (timestamp, region_id, variable, value_mwh).",
              file=sys.stderr)
        return 1

    pdf = df_out.toPandas()
    _write_parquet_to_silver(pdf, silver_endpoint, silver_access, silver_secret,
                             silver_bucket, silver_key)
    print(f"Wrote {row_count:,} rows to s3://{silver_bucket}/{silver_key}", file=sys.stderr)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
