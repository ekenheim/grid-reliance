"""
Synthetic weather and grid data generator for 8 Nordic zones × 5 years hourly.

Uses Matern spatial covariance matrix (great-circle distances) to generate
correlated wind speeds. Injects 5-10 correlated drought events per year and
20-30 price spike events (>200 EUR/MWh) that correlate with wind shortfalls.

Output: PostgreSQL partitioned tables (weather_obs, grid_demand, spot_prices).
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env from repo root (parent of generator/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from typing import Any

# Zone list: 8 Nordic zones (SE1-SE4, NO1-NO5 subset, DK1-DK2, FI)
NORDIC_ZONES = [
    "SE1", "SE2", "SE3", "SE4",
    "NO1", "NO2",
    "DK1", "DK2",
    "FI",
]

# Zone centroids (lat, lon) for great-circle distance
ZONE_CENTROIDS = {
    "SE1": (62.0, 14.5),
    "SE2": (62.5, 17.5),
    "SE3": (59.5, 18.0),
    "SE4": (56.0, 14.0),
    "NO1": (59.5, 10.5),
    "NO2": (64.0, 11.5),
    "DK1": (56.5, 9.5),
    "DK2": (55.5, 12.0),
    "FI": (61.5, 25.0),
}

# Years of hourly data
START_YEAR = 2019
END_YEAR = 2024
HOURS_PER_YEAR = 8760


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def build_matern_covariance_matrix(
    zone_coords: dict[str, tuple[float, float]],
    zone_ids: list[str],
    lengthscale_km: float = 100.0,
    nu: float = 1.5,
    sigma2: float = 1.0,
) -> np.ndarray:
    """
    Build spatial covariance matrix using Matern kernel with great-circle distances.

    Args:
        zone_coords: Dict of zone_id -> (lat, lon).
        zone_ids: Order of zones (rows/cols of matrix).
        lengthscale_km: Spatial lengthscale in km.
        nu: Matern smoothness (1.5 = Matern 3/2).
        sigma2: Marginal variance.

    Returns:
        Covariance matrix (numpy array).
    """
    n = len(zone_ids)
    coords = np.array([zone_coords[z] for z in zone_ids])
    # Pairwise great-circle distances (km)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                d = _haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                D[i, j] = D[j, i] = d
    # Matern 3/2: K(r) = sigma2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
    r = D / (lengthscale_km + 1e-9)
    if nu == 1.5:
        K = sigma2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    else:
        from scipy.special import kv
        r = np.maximum(r, 1e-10)
        sqrt_nu = np.sqrt(2 * nu) * r
        K = sigma2 * (2 ** (1 - nu) / __import__("math").gamma(nu)) * (sqrt_nu ** nu) * kv(nu, sqrt_nu)
    np.fill_diagonal(K, sigma2)
    return K.astype(np.float64)


def inject_correlated_drought_events(
    wind_speeds: np.ndarray,
    timestamps: list[datetime],
    zone_ids: list[str],
    events_per_year: int = 5,
    scale: float = 0.4,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Inject 5-10 correlated drought events per year where multiple regions
    simultaneously experience low wind.

    Args:
        wind_speeds: Array of shape (n_timesteps, n_zones).
        timestamps: List of timestamps.
        zone_ids: List of zone IDs.
        events_per_year: Number of drought events per year.
        scale: Multiplier for wind during drought (e.g. 0.4 = 40% of original).
        seed: Random seed.

    Returns:
        Modified wind_speeds array.
    """
    rng = random.Random(seed)
    n_timesteps, n_zones = wind_speeds.shape
    years = sorted(set(t.year for t in timestamps))
    out = wind_speeds.copy()
    for year in years:
        n_events = rng.randint(5, min(11, events_per_year + 6))
        for _ in range(n_events):
            t_idx = rng.randint(0, n_timesteps - 1)
            n_affected = rng.randint(2, max(2, n_zones))
            affected = set(rng.sample(range(n_zones), n_affected))
            for j in affected:
                out[t_idx, j] *= scale
    return out


def generate_price_spikes(
    timestamps: list[datetime],
    zone_ids: list[str],
    drought_timesteps: set[int],
    n_spikes: int = 25,
    spike_price_min: float = 200.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate 20-30 price spike events (>200 EUR/MWh) that correlate with
    wind shortfalls (drought_timesteps).

    Returns:
        DataFrame with columns: timestamp, region_id, price_eur_mwh.
    """
    rng = random.Random(seed)
    rows = []
    base_price = 50.0
    drought_list = sorted(drought_timesteps)
    for t_idx, ts in enumerate(timestamps):
        if drought_list and t_idx in drought_timesteps:
            price = base_price + rng.uniform(spike_price_min, 400.0) if rng.random() < 0.7 else base_price + rng.uniform(0, 100)
        else:
            price = base_price + rng.uniform(0, 80)
        for z in zone_ids:
            rows.append({"timestamp": ts, "region_id": z, "price_eur_mwh": round(price, 2)})
    return pd.DataFrame(rows)


def create_postgres_schema(engine: Any) -> None:
    """
    Create PostgreSQL partitioned tables for weather_obs, grid_demand, spot_prices.
    Drops existing tables so re-runs replace data.
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        for table in ("weather_obs", "grid_demand", "spot_prices"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.commit()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE weather_obs (
                timestamp TIMESTAMPTZ NOT NULL,
                region_id TEXT NOT NULL,
                wind_speed_mps REAL
            ) PARTITION BY RANGE (timestamp);
        """))
        conn.execute(text("""
            CREATE TABLE grid_demand (
                timestamp TIMESTAMPTZ NOT NULL,
                region_id TEXT NOT NULL,
                demand_mw REAL
            ) PARTITION BY RANGE (timestamp);
        """))
        conn.execute(text("""
            CREATE TABLE spot_prices (
                timestamp TIMESTAMPTZ NOT NULL,
                region_id TEXT NOT NULL,
                price_eur_mwh REAL
            ) PARTITION BY RANGE (timestamp);
        """))
        conn.commit()
    with engine.connect() as conn:
        for year in range(START_YEAR, END_YEAR + 1):
            for name in ("weather_obs", "grid_demand", "spot_prices"):
                part = f"{name}_{year}"
                conn.execute(text(f"""
                    CREATE TABLE {part}
                    PARTITION OF {name}
                    FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01');
                """))
        conn.commit()


def bulk_load_to_postgres(
    weather_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    connection_string: str,
    chunksize: int = 50_000,
) -> None:
    """
    Bulk-load generated DataFrames into PostgreSQL partitioned tables.
    """
    from sqlalchemy import create_engine
    engine = create_engine(connection_string)
    if not weather_df.empty:
        weather_df.to_sql(
            "weather_obs",
            engine,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )
    if not demand_df.empty:
        demand_df.to_sql(
            "grid_demand",
            engine,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )
    if not prices_df.empty:
        prices_df.to_sql(
            "spot_prices",
            engine,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )


def main() -> None:
    """Generate synthetic data and load to PostgreSQL."""
    from sqlalchemy import create_engine
    db_url = os.getenv("DATABASE_URL", "postgresql://grid_resilience:grid_resilience@localhost:5432/grid_resilience")
    seed = int(os.environ.get("GENERATOR_SEED", "42"))
    np.random.seed(seed)
    random.seed(seed)

    zone_ids = NORDIC_ZONES
    n_zones = len(zone_ids)
    timestamps = []
    t = datetime(START_YEAR, 1, 1, 0, 0, 0)
    end = datetime(END_YEAR, 12, 31, 23, 0, 0)
    while t <= end:
        timestamps.append(t)
        t += timedelta(hours=1)
    n_timesteps = len(timestamps)

    K = build_matern_covariance_matrix(
        ZONE_CENTROIDS,
        zone_ids,
        lengthscale_km=100.0,
        nu=1.5,
        sigma2=4.0,
    )
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n_zones))
    # (n_timesteps, n_zones) from N(0, I) then L @ z
    z = np.random.randn(n_timesteps, n_zones).astype(np.float64)
    wind_speeds = (L @ z.T).T
    mean_wind = 7.0
    wind_speeds = mean_wind + wind_speeds
    wind_speeds = np.maximum(wind_speeds, 0.0)
    wind_speeds = inject_correlated_drought_events(
        wind_speeds, timestamps, zone_ids, events_per_year=5, seed=seed
    )
    drought_timesteps = set()
    for t_idx in range(n_timesteps):
        if np.any(wind_speeds[t_idx] < 3.0):
            drought_timesteps.add(t_idx)
    weather_rows = []
    for t_idx, ts in enumerate(timestamps):
        for j, zid in enumerate(zone_ids):
            weather_rows.append({
                "timestamp": ts,
                "region_id": zid,
                "wind_speed_mps": round(float(wind_speeds[t_idx, j]), 4),
            })
    weather_df = pd.DataFrame(weather_rows)
    demand_df = pd.DataFrame(columns=["timestamp", "region_id", "demand_mw"])
    prices_df = generate_price_spikes(timestamps, zone_ids, drought_timesteps, n_spikes=25, seed=seed)
    engine = create_engine(db_url)
    create_postgres_schema(engine)
    bulk_load_to_postgres(weather_df, demand_df, prices_df, db_url)
    print(f"Loaded {len(weather_df)} weather rows, {len(prices_df)} price rows.")


if __name__ == "__main__":
    main()
