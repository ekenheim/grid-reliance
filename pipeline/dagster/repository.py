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
    gold_io_manager,
    gold_pickle_io_manager,
    noop_io_manager,
)
from pipeline.dagster.jobs import (
    grid_resilience_full_pipeline_job,
    grid_resilience_retrain_job,
)
from pipeline.dagster.resources import silver_resource, gold_resource, postgres_resource, redpanda_resource
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
        "silver":                silver_resource,
        "gold":                  gold_resource,
        "postgres":              postgres_resource,
        "gold_io_manager":       gold_io_manager,
        "gold_pickle_io_manager": gold_pickle_io_manager,
        "noop_io_manager":       noop_io_manager,
        "redpanda":              redpanda_resource,
    },
)
