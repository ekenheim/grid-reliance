"""
Dagster asset definitions for the Grid Resilience HSGP pipeline.

Asset graph:
  raw_weather_obs -> grid_snapshots -> hsgp_model -> tail_risk_forecasts -> risk_alerts
"""

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from dagster import asset, AssetIn, Output, MetadataValue

logger = logging.getLogger(__name__)

HSGP_IDATA_KEY = "dagster/hsgp_model_idata.nc"
TEMPORAL_SCALE_H = 168.0


@asset(
    description=(
        "Raw weather observations. Primary source: Rook Silver bucket "
        "(silver/grid_snapshots.parquet written by Spark ERA5/ENTSO-E ingest). "
        "Falls back to Postgres weather_obs when Silver is not configured or the key is absent."
    ),
    io_manager_key="gold_io_manager",
    required_resource_keys={"silver", "postgres"},
)
def raw_weather_obs(context) -> Output[pd.DataFrame]:
    """Load raw weather observations — Silver bucket first, Postgres fallback."""
    silver = context.resources.silver

    if silver is not None:
        try:
            response = silver["client"].get_object(
                Bucket=silver["bucket"], Key="silver/grid_snapshots.parquet"
            )
            df = pd.read_parquet(io.BytesIO(response["Body"].read()))
            if not df.empty:
                context.log.info("Loaded %d rows from Silver bucket.", len(df))
                return Output(df, metadata={"num_rows": len(df), "source": "rook-silver"})
            context.log.warning("Silver key returned empty Parquet; falling back to Postgres.")
        except Exception as exc:
            context.log.warning(
                "Silver read failed (%s); falling back to Postgres weather_obs table.", exc
            )

    # Postgres fallback — used during local dev and before Spark ingest is deployed
    from sqlalchemy import create_engine, text
    conn_string = context.resources.postgres
    engine = create_engine(conn_string)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT timestamp, region_id, wind_speed_mps FROM weather_obs ORDER BY timestamp, region_id"),
            conn,
        )
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "region_id", "wind_speed_mps"])
    context.log.info("Loaded %d rows from Postgres weather_obs (fallback).", len(df))
    return Output(df, metadata={"num_rows": len(df), "source": "postgres-fallback"})


@asset(
    ins={"raw_weather_obs": AssetIn("raw_weather_obs")},
    description="1-hour tumbling window snapshots: all 8 regions aggregated per timestep.",
    io_manager_key="gold_io_manager",
    required_resource_keys={"gold"},
)
def grid_snapshots(context, raw_weather_obs: pd.DataFrame) -> Output[pd.DataFrame]:
    """Spatial snapshots for HSGP input. One row per (timestamp, region_id); 1h resolution."""
    if raw_weather_obs.empty:
        df = pd.DataFrame(columns=["timestamp", "region_id", "wind_speed_mps"])
    else:
        df = raw_weather_obs.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.groupby(["timestamp", "region_id"], as_index=False).agg({"wind_speed_mps": "mean"})
    return Output(
        df,
        metadata={"num_rows": len(df), "source": "aggregated"},
    )


@asset(
    ins={"grid_snapshots": AssetIn("grid_snapshots")},
    description="Trained HSGP model (PyMC + Nutpie) with Matern 3/2 spatial + 5/2 temporal kernel.",
    io_manager_key="gold_pickle_io_manager",
    required_resource_keys={"gold"},
)
def hsgp_model(context, grid_snapshots: pd.DataFrame) -> Output[dict]:
    """Train HSGP with MCMC (Nutpie). Store idata in Rook Gold and metadata in pickle."""
    from pipeline.model.hsgp_model import train_hsgp
    from pipeline.model.diagnostics import run_diagnostics

    result = train_hsgp(
        grid_snapshots,
        m_spatial=5,
        m_temporal=10,
        draws=300,
        tune=200,
        chains=2,
        random_seed=42,
        idata_path=None,
    )
    if result.get("idata") is not None:
        buf = io.BytesIO()
        result["idata"].to_netcdf(buf)
        buf.seek(0)
        gold = context.resources.gold
        gold["client"].put_object(
            Bucket=gold["bucket"],
            Key=HSGP_IDATA_KEY,
            Body=buf.getvalue(),
        )
        result["idata_path"] = HSGP_IDATA_KEY
        diagnostics = run_diagnostics(None, idata=result["idata"])
        metadata_extra = {"n_divergences": diagnostics.get("n_divergences"), "max_r_hat": diagnostics.get("max_r_hat")}
    else:
        result["idata_path"] = None
        metadata_extra = {}
    out = {
        "idata_path": result.get("idata_path"),
        "metadata": result.get("metadata", {}),
        "spatial_correlation": result.get("spatial_correlation"),
        "m_spatial": 5,
        "m_temporal": 10,
    }
    return Output(
        out,
        metadata={"model_type": "HSGP", "status": "trained", **metadata_extra},
    )


@asset(
    ins={
        "hsgp_model": AssetIn("hsgp_model"),
        "grid_snapshots": AssetIn("grid_snapshots"),
    },
    description="Tail-risk forecasts: P(shortfall > threshold) for 24/48/72h ahead.",
    io_manager_key="gold_io_manager",
    required_resource_keys={"gold"},
)
def tail_risk_forecasts(context, hsgp_model: dict, grid_snapshots: pd.DataFrame) -> Output[pd.DataFrame]:
    """Generate probabilistic tail-risk forecasts from HSGP posterior predictive."""
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    from pipeline.model.hsgp_model import build_hsgp_model, sample_posterior_predictive, compute_tail_risk
    import arviz as az

    idata_path = hsgp_model.get("idata_path")
    if not idata_path or grid_snapshots.empty:
        return Output(
            pd.DataFrame(columns=["timestamp", "region_id", "horizon_h", "p_shortfall"]),
            metadata={"num_rows": 0, "source": "skip"},
        )
    gold = context.resources.gold
    resp = gold["client"].get_object(Bucket=gold["bucket"], Key=idata_path)
    idata = az.from_netcdf(io.BytesIO(resp["Body"].read()))
    meta = hsgp_model.get("metadata", {})
    zone_ids = meta.get("zone_ids", [])
    if not zone_ids and "region_id" in grid_snapshots.columns:
        zone_ids = sorted(grid_snapshots["region_id"].unique().tolist())
    m_spatial = hsgp_model.get("m_spatial", 5)
    m_temporal = hsgp_model.get("m_temporal", 10)
    X_full, y_full, _ = prepare_hsgp_2d_input(grid_snapshots, temporal_scale_h=TEMPORAL_SCALE_H)
    model, gp = build_hsgp_model(X_full, y_full, m_spatial=m_spatial, m_temporal=m_temporal)
    result_tmp = {"idata": idata, "model": model, "gp": gp}

    grid_snapshots = grid_snapshots.copy()
    grid_snapshots["timestamp"] = pd.to_datetime(grid_snapshots["timestamp"])
    t_max = grid_snapshots["timestamp"].max()
    t_min = grid_snapshots["timestamp"].min()
    base_hours = (t_max - t_min).total_seconds() / 3600.0
    n_zones = len(zone_ids)
    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}
    rows = []
    for horizon_h in (24, 48, 72):
        t_new = t_max + pd.Timedelta(hours=horizon_h)
        temporal_norm = (base_hours + horizon_h) / TEMPORAL_SCALE_H
        spatial_coords_new = np.array([zone_to_idx[z] / max(n_zones - 1, 1) for z in zone_ids], dtype=np.float64)
        temporal_coords_new = np.full(n_zones, temporal_norm, dtype=np.float64)
        samples = sample_posterior_predictive(result_tmp, spatial_coords_new, temporal_coords_new, n_samples=500)
        p_shortfall = compute_tail_risk(samples, threshold_mps=3.0)
        for i, zid in enumerate(zone_ids):
            rows.append({
                "timestamp": t_new,
                "region_id": zid,
                "horizon_h": horizon_h,
                "p_shortfall": float(p_shortfall[i]),
            })
    df = pd.DataFrame(rows)
    return Output(
        df,
        metadata={"num_rows": len(df), "source": "hsgp_ppc"},
    )


@asset(
    ins={"tail_risk_forecasts": AssetIn("tail_risk_forecasts")},
    description="Risk alerts when P(correlated drought) > threshold. Phase 2: publish to Redpanda.",
    io_manager_key="noop_io_manager",
)
def risk_alerts(context, tail_risk_forecasts: pd.DataFrame) -> Output[None]:
    """Log or publish alerts when p_shortfall exceeds threshold (e.g. 0.9)."""
    threshold = 0.9
    if tail_risk_forecasts.empty or "p_shortfall" not in tail_risk_forecasts.columns:
        return Output(None, metadata={"alerts_published": MetadataValue.int(0)})
    alerts = tail_risk_forecasts[tail_risk_forecasts["p_shortfall"] >= threshold]
    n = len(alerts)
    for _, row in alerts.iterrows():
        context.log.warning(
            "Risk alert: region=%s horizon_h=%s p_shortfall=%.3f",
            row.get("region_id"), row.get("horizon_h"), row.get("p_shortfall"),
        )
    return Output(
        None,
        metadata={"alerts_published": MetadataValue.int(n)},
    )
