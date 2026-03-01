"""
Dagster IO managers for MinIO (Parquet, Pickle) and PostgreSQL (tables).
"""

import io
import logging
import pickle

import pandas as pd
from dagster import IOManager, io_manager, InputContext, OutputContext
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class MinIOParquetIOManager(IOManager):
    """Read/write Pandas DataFrames as Parquet files in MinIO."""

    def __init__(self, s3_client, bucket: str, prefix: str = "dagster"):
        self._client = s3_client
        self._bucket = bucket
        self._prefix = prefix

    def _key(self, context) -> str:
        asset_key = "/".join(context.asset_key.path)
        return f"{self._prefix}/{asset_key}.parquet"

    def handle_output(self, context: OutputContext, obj):
        if obj is None:
            return
        key = self._key(context)
        buf = io.BytesIO()
        obj.to_parquet(buf, index=False)
        buf.seek(0)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=buf)
        context.log.info(f"Wrote {len(obj)} rows to s3://{self._bucket}/{key}")

    def load_input(self, context: InputContext):
        key = self._key(context)
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        buf = io.BytesIO(response["Body"].read())
        df = pd.read_parquet(buf)
        context.log.info(f"Read {len(df)} rows from s3://{self._bucket}/{key}")
        return df


class PostgresIOManager(IOManager):
    """Read/write Pandas DataFrames as PostgreSQL tables."""

    def __init__(self, connection_string: str, schema: str = "public"):
        self._connection_string = connection_string
        self._schema = schema

    def _table_name(self, context) -> str:
        return "_".join(context.asset_key.path)

    def handle_output(self, context: OutputContext, obj):
        if obj is None:
            return
        table = self._table_name(context)
        engine = create_engine(self._connection_string)
        obj.to_sql(table, engine, schema=self._schema, if_exists="replace", index=False)
        context.log.info(f"Wrote {len(obj)} rows to {self._schema}.{table}")

    def load_input(self, context: InputContext):
        table = self._table_name(context)
        engine = create_engine(self._connection_string)
        df = pd.read_sql_table(table, engine, schema=self._schema)
        context.log.info(f"Read {len(df)} rows from {self._schema}.{table}")
        return df


class MinIOPickleIOManager(IOManager):
    """Read/write arbitrary Python objects as pickle files in MinIO."""

    def __init__(self, s3_client, bucket: str, prefix: str = "dagster"):
        self._client = s3_client
        self._bucket = bucket
        self._prefix = prefix

    def _key(self, context) -> str:
        asset_key = "/".join(context.asset_key.path)
        return f"{self._prefix}/{asset_key}.pkl"

    def handle_output(self, context: OutputContext, obj):
        if obj is None:
            return
        key = self._key(context)
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        buf.seek(0)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=buf)
        context.log.info(f"Pickled output to s3://{self._bucket}/{key}")

    def load_input(self, context: InputContext):
        key = self._key(context)
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        obj = pickle.loads(response["Body"].read())
        context.log.info(f"Loaded pickle from s3://{self._bucket}/{key}")
        return obj


class NoOpIOManager(IOManager):
    """For assets that write to the filesystem directly and need no persistence."""

    def handle_output(self, context, obj):
        pass

    def load_input(self, context):
        return None


@io_manager
def noop_io_manager(context):
    return NoOpIOManager()


@io_manager(required_resource_keys={"gold"})
def gold_io_manager(context):
    gold = context.resources.gold
    return MinIOParquetIOManager(gold["client"], gold["bucket"])


@io_manager(required_resource_keys={"gold"})
def gold_pickle_io_manager(context):
    gold = context.resources.gold
    return MinIOPickleIOManager(gold["client"], gold["bucket"])


# --- Local dev / legacy fallbacks (not registered in production Definitions) ---

@io_manager(required_resource_keys={"minio"})
def minio_io_manager(context):
    minio = context.resources.minio
    return MinIOParquetIOManager(minio["client"], minio["bucket"])


@io_manager(required_resource_keys={"minio"})
def minio_pickle_io_manager(context):
    minio = context.resources.minio
    return MinIOPickleIOManager(minio["client"], minio["bucket"])


@io_manager(required_resource_keys={"postgres"})
def postgres_io_manager(context):
    conn_string = context.resources.postgres
    return PostgresIOManager(conn_string)
