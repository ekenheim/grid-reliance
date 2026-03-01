"""
Dagster job definitions for the Grid Resilience pipeline.
"""

from dagster import define_asset_job, AssetSelection

from pipeline.dagster.assets import (
    fetch_era5_bronze,
    fetch_entsoe_bronze,
    raw_weather_obs,
    grid_snapshots,
    hsgp_model,
    tail_risk_forecasts,
    risk_alerts,
)

grid_resilience_bronze_seed_job = define_asset_job(
    name="grid_resilience_bronze_seed",
    selection=AssetSelection.assets(fetch_era5_bronze, fetch_entsoe_bronze),
    description=(
        "One-shot Bronze seeding: fetch ERA5 NetCDF + ENTSO-E CSVs and upload to Bronze. "
        "Trigger this before the Spark ingest jobs to populate Bronze for the first time. "
        "Date range via ERA5_FETCH_START/END and ENTSOE_FETCH_START/END env vars."
    ),
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
