"""
Hilbert Space GP (HSGP) model for spatio-temporal weather covariance.

Uses separable kernel: spatial Matern 3/2 × temporal Matern 5/2.
HSGP approximation reduces inference from O(N^3) to O(N*M^2) for tractable
training on 5 years of hourly data (8 Nordic zones).

MCMC with Nutpie for proper posterior and tail-risk; SVI/NumPyro optional in Phase 2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# M basis functions: start with M=50 (5 spatial × 10 temporal), increase if needed
DEFAULT_M_SPATIAL = 5
DEFAULT_M_TEMPORAL = 10


def build_hsgp_model(
    X: np.ndarray,
    y: np.ndarray,
    m_spatial: int = DEFAULT_M_SPATIAL,
    m_temporal: int = DEFAULT_M_TEMPORAL,
    coords: dict | None = None,
) -> Any:
    """
    Build PyMC HSGP model with separable Matern kernel.

    Args:
        X: (n_obs, 2) - col 0: normalized spatial, col 1: normalized temporal.
        y: (n_obs,) - observed wind speed (m/s).
        m_spatial: Number of spatial basis functions.
        m_temporal: Number of temporal basis functions.
        coords: Optional coords dict for InferenceData (e.g. {"obs": np.arange(n_obs)}).

    Returns:
        Tuple of (pm.Model, gp.HSGP).
    """
    import pymc as pm

    n_obs = X.shape[0]
    if coords is None:
        coords = {"obs": np.arange(n_obs)}

    with pm.Model(coords=coords) as model:
        # Data containers for out-of-sample prediction
        X_data = pm.Data("X", X)
        y_data = pm.Data("y", y)

        # Priors: lengthscales per dimension (normalized inputs; separable via ARD)
        ell_spatial = pm.InverseGamma("ell_spatial", alpha=3.0, beta=0.5)
        ell_temporal = pm.InverseGamma("ell_temporal", alpha=3.0, beta=1.0)
        eta = pm.HalfNormal("eta", sigma=3.0)
        sigma = pm.HalfNormal("sigma", sigma=2.0)

        # Single 2D kernel with per-dimension lengthscales (separable ARD; HSGP PSD requires single kernel)
        cov = eta**2 * pm.gp.cov.Matern52(2, ls=[ell_spatial, ell_temporal])

        gp = pm.gp.HSGP(m=[m_spatial, m_temporal], c=1.5, cov_func=cov)
        f = gp.prior("f", X=X_data)

        pm.Normal("y_obs", mu=f, sigma=sigma, observed=y_data, dims="obs")

    return model, gp


def train_hsgp(
    grid_snapshots: pd.DataFrame,
    m_spatial: int = DEFAULT_M_SPATIAL,
    m_temporal: int = DEFAULT_M_TEMPORAL,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
    idata_path: str | None = None,
) -> dict:
    """
    Train HSGP with MCMC (Nutpie NUTS).

    Args:
        grid_snapshots: DataFrame with columns timestamp, region_id, wind_speed_mps.
        m_spatial: Spatial basis functions.
        m_temporal: Temporal basis functions.
        draws: Number of posterior draws per chain.
        tune: Number of tuning steps per chain.
        chains: Number of chains.
        random_seed: Random seed.
        idata_path: If set, save InferenceData to this path (NetCDF) immediately after sampling.

    Returns:
        Dict with idata (or path), model (reference), metadata, spatial_correlation (optional).
    """
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    import pymc as pm

    if grid_snapshots.empty or len(grid_snapshots) < 10:
        logger.warning("grid_snapshots empty or too small; returning stub result")
        return {"idata": None, "model": None, "gp": None, "metadata": {}, "spatial_correlation": None}

    X, y, metadata = prepare_hsgp_2d_input(grid_snapshots)
    n_obs = X.shape[0]
    coords = {"obs": np.arange(n_obs)}
    model, gp = build_hsgp_model(X, y, m_spatial=m_spatial, m_temporal=m_temporal, coords=coords)

    try:
        import nutpie  # noqa: F401
        nuts_sampler = "nutpie"
    except ImportError:
        nuts_sampler = "pymc"
        logger.warning(
            "nutpie not installed; using default PyMC NUTS (slower). "
            "Install with: pip install nutpie  (or conda install -c conda-forge nutpie)"
        )

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            idata_kwargs={"log_likelihood": True},
        )
        if idata_path:
            idata.to_netcdf(idata_path)
            logger.info("Saved idata to %s", idata_path)
        try:
            pm.compute_log_likelihood(idata, model=model)
        except Exception as e:
            logger.debug("log_likelihood computation skipped: %s", e)

    spatial_correlation = None
    try:
        ell_s = idata.posterior["ell_spatial"].values.flatten()
        ell_t = idata.posterior["ell_temporal"].values.flatten()
        eta = idata.posterior["eta"].values.flatten()
        spatial_correlation = np.corrcoef(np.column_stack([ell_s, ell_t, eta])) if len(ell_s) > 1 else None
    except Exception as e:
        logger.debug("Could not derive spatial correlation: %s", e)

    return {
        "idata": idata,
        "model": model,
        "gp": gp,
        "metadata": metadata,
        "spatial_correlation": spatial_correlation,
        "idata_path": idata_path,
    }


def sample_posterior_predictive(
    result: dict,
    spatial_coords_new: np.ndarray,
    temporal_coords_new: np.ndarray,
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Sample from posterior predictive for tail-risk computation.

    Args:
        result: Dict from train_hsgp with keys idata, model, gp.
        spatial_coords_new: (n_new,) normalized spatial coords (same scale as training).
        temporal_coords_new: (n_new,) normalized temporal coords (same scale as training).
        n_samples: Number of samples to return.

    Returns:
        (n_samples, n_new) array of wind speed samples.
    """
    import pymc as pm

    idata = result.get("idata")
    model = result.get("model")
    gp = result.get("gp")
    if idata is None or model is None or gp is None:
        n_new = len(spatial_coords_new)
        return np.full((n_samples, n_new), np.nan)

    X_new = np.column_stack([
        np.asarray(spatial_coords_new, dtype=np.float64).ravel(),
        np.asarray(temporal_coords_new, dtype=np.float64).ravel(),
    ])
    n_new = X_new.shape[0]

    with model:
        fcond = gp.conditional("fcond", Xnew=X_new)
        sigma = model["sigma"]
        y_new = pm.Normal("y_new", mu=fcond, sigma=sigma, shape=(n_new,))
        pp = pm.sample_posterior_predictive(
            idata,
            var_names=["y_new"],
            predictions=True,
            random_seed=42,
        )
    samples = pp.posterior_predictive["y_new"].values
    samples = np.reshape(samples, (-1, n_new))
    if samples.shape[0] > n_samples:
        samples = samples[:n_samples]
    elif samples.shape[0] < n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(samples.shape[0], size=n_samples, replace=True)
        samples = samples[idx]
    return samples


def compute_tail_risk(
    samples: np.ndarray,
    threshold_mps: float = 3.0,
) -> np.ndarray:
    """
    P(shortfall > threshold) = fraction of samples below threshold.

    Returns:
        (n_new,) array of probabilities.
    """
    return np.mean(samples < threshold_mps, axis=0)
