"""
Dagster code location / repository definition for the Grid Resilience pipeline.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from dagster import Definitions

from pipeline.dagster.assets import (
    raw_weather_obs,
    grid_snapshots,
    hsgp_model,
    tail_risk_forecasts,
    risk_alerts,
)
from pipeline.dagster.io_managers import (
    minio_io_manager,
    minio_pickle_io_manager,
    postgres_io_manager,
    noop_io_manager,
)
from pipeline.dagster.jobs import (
    grid_resilience_full_pipeline_job,
    grid_resilience_retrain_job,
)
from pipeline.dagster.resources import minio_resource, postgres_resource, redpanda_resource
from pipeline.dagster.schedules import retrain_schedule, full_pipeline_schedule

defs = Definitions(
    assets=[
        raw_weather_obs,
        grid_snapshots,
        hsgp_model,
        tail_risk_forecasts,
        risk_alerts,
    ],
    jobs=[grid_resilience_full_pipeline_job, grid_resilience_retrain_job],
    schedules=[retrain_schedule, full_pipeline_schedule],
    resources={
        "minio": minio_resource,
        "postgres": postgres_resource,
        "redpanda": redpanda_resource,
        "minio_io_manager": minio_io_manager,
        "minio_pickle_io_manager": minio_pickle_io_manager,
        "postgres_io_manager": postgres_io_manager,
        "noop_io_manager": noop_io_manager,
    },
)
