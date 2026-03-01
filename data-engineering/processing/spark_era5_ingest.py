"""
Minimal Spark job: read from Bronze (S3), write grid_snapshots-style Parquet to Silver.

Expects env (from Rook ObjectBucketClaim or equivalent):
  BUCKET_HOST, BUCKET_PORT, BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  BRONZE_BUCKET_NAME (and optionally BRONZE_BUCKET_HOST/PORT if different)

Input path: s3a://BRONZE_BUCKET_NAME/... (e.g. era5/ or raw/)
Output path: s3a://BUCKET_NAME/silver/grid_snapshots.parquet

Schema written: timestamp, region_id, wind_speed_mps (see CONTRACT.md).
This stub reads whatever Parquet exists in Bronze with compatible columns, or generates
a tiny synthetic row set so the job runs end-to-end; replace with real ERA5/ENTSO-E ETL.
"""

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_endpoint():
    host = os.environ.get("BUCKET_HOST", "").strip()
    port = os.environ.get("BUCKET_PORT", "80").strip()
    if not host:
        return None
    return f"http://{host}:{port}"


def main():
    endpoint = get_endpoint()
    bronze_bucket = os.environ.get("BRONZE_BUCKET_NAME", "grid-resilience-bronze").strip()
    silver_bucket = os.environ.get("BUCKET_NAME", "grid-resilience-silver").strip()
    if not endpoint:
        print("BUCKET_HOST not set; cannot configure S3.", file=sys.stderr)
        sys.exit(1)

    spark = (
        SparkSession.builder.appName("grid-resilience-era5-ingest")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", ""))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    bronze_prefix = os.environ.get("BRONZE_PREFIX", "era5").strip().rstrip("/")
    bronze_path = f"s3a://{bronze_bucket}/{bronze_prefix}/"
    silver_path = f"s3a://{silver_bucket}/silver/grid_snapshots.parquet"

    try:
        df = spark.read.parquet(bronze_path)
    except Exception:
        # No Parquet in Bronze yet: write a minimal stub so the job completes (replace with real ETL)
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
        schema = StructType([
            StructField("timestamp", TimestampType()),
            StructField("region_id", StringType()),
            StructField("wind_speed_mps", DoubleType()),
        ])
        rows = [
            (datetime(2024, 1, 1, 0, 0, 0), "SE1", 8.0),
            (datetime(2024, 1, 1, 0, 0, 0), "DK1", 7.0),
        ]
        df = spark.createDataFrame(rows, schema=schema)

    if "wind_speed_mps" not in df.columns and "wind_speed" in df.columns:
        df = df.withColumnRenamed("wind_speed", "wind_speed_mps")
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.withColumnRenamed("time", "timestamp")
    df = df.select(
        F.col("timestamp"),
        F.col("region_id"),
        F.col("wind_speed_mps") if "wind_speed_mps" in df.columns else F.lit(0.0).alias("wind_speed_mps"),
    )
    df.write.mode("overwrite").parquet(silver_path)
    print(f"Wrote {silver_path}", file=sys.stderr)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
