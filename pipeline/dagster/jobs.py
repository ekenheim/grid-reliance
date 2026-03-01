"""
Dagster job definitions for the Grid Resilience pipeline.
"""

from dagster import define_asset_job, AssetSelection

from pipeline.dagster.assets import (
    raw_weather_obs,
    grid_snapshots,
    hsgp_model,
    tail_risk_forecasts,
    risk_alerts,
)

grid_resilience_full_pipeline_job = define_asset_job(
    name="grid_resilience_full_pipeline",
    selection=AssetSelection.assets(
        raw_weather_obs,
        grid_snapshots,
        hsgp_model,
        tail_risk_forecasts,
        risk_alerts,
    ),
    description="Run full pipeline: ingest -> snapshots -> train HSGP -> forecast -> publish alerts",
)

grid_resilience_retrain_job = define_asset_job(
    name="grid_resilience_retrain",
    selection=AssetSelection.assets(
        raw_weather_obs,
        grid_snapshots,
        hsgp_model,
        tail_risk_forecasts,
        risk_alerts,
    ),
    description="6-hourly retrain: ingest -> train -> forecast -> alert",
)
