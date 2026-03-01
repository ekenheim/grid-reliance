"""
Feature engineering for Grid Resilience HSGP pipeline.

Prepares spatial coordinates (great-circle distances), temporal coordinates,
and wind speed arrays from raw weather/grid snapshots.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nordic zone centroids (lat, lon) for great-circle distance computation
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def build_spatial_distance_matrix(zone_ids: list[str]) -> np.ndarray:
    """
    Build pairwise great-circle distance matrix (km) for zone centroids.

    Returns:
        (n_zones, n_zones) symmetric matrix.
    """
    n = len(zone_ids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = ZONE_CENTROIDS.get(zone_ids[i], (0, 0))
            lat2, lon2 = ZONE_CENTROIDS.get(zone_ids[j], (0, 0))
            d = haversine_km(lat1, lon1, lat2, lon2)
            D[i, j] = D[j, i] = d
    return D


def prepare_hsgp_input(
    df: pd.DataFrame,
    region_col: str = "region_id",
    timestamp_col: str = "timestamp",
    value_col: str = "wind_speed_mps",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Prepare arrays for HSGP model from grid snapshots DataFrame.

    Returns:
        spatial_coords: (n_obs,) - region index
        temporal_coords: (n_obs,) - numeric hours since epoch
        values: (n_obs,) - wind speed
        metadata: dict with zone_ids, n_zones, n_timesteps
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["_temporal"] = (df[timestamp_col] - df[timestamp_col].min()).dt.total_seconds() / 3600.0

    zone_ids = sorted(df[region_col].unique().tolist())
    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}
    df["_spatial_idx"] = df[region_col].map(zone_to_idx)

    spatial_coords = df["_spatial_idx"].values.astype(np.float64)
    temporal_coords = df["_temporal"].values.astype(np.float64)
    values = df[value_col].values.astype(np.float64)

    metadata = {
        "zone_ids": zone_ids,
        "n_zones": len(zone_ids),
        "n_timesteps": df[timestamp_col].nunique(),
    }
    return spatial_coords, temporal_coords, values, metadata


def prepare_hsgp_2d_input(
    df: pd.DataFrame,
    region_col: str = "region_id",
    timestamp_col: str = "timestamp",
    value_col: str = "wind_speed_mps",
    temporal_scale_h: float = 168.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare 2D input X for HSGP (spatial and temporal dimensions).

    Returns:
        X: (n_obs, 2) - col 0: normalized spatial coord [0,1], col 1: normalized temporal (hours / temporal_scale_h)
        y: (n_obs,) - wind speed
        metadata: dict with zone_ids, n_zones, n_timesteps
    """
    spatial_coords, temporal_coords, values, metadata = prepare_hsgp_input(
        df, region_col=region_col, timestamp_col=timestamp_col, value_col=value_col
    )
    n_zones = metadata["n_zones"]
    spatial_norm = spatial_coords / max(n_zones - 1, 1)
    temporal_norm = temporal_coords / temporal_scale_h
    X = np.column_stack([spatial_norm, temporal_norm]).astype(np.float64)
    return X, values, metadata
