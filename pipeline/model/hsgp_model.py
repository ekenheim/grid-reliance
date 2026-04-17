"""
Hilbert Space GP (HSGP) model for spatio-temporal weather covariance.

Uses separable kernel: spatial Matern 3/2 × temporal Matern 5/2.
HSGP approximation reduces inference from O(N^3) to O(N*M^2) for tractable
training on 5 years of hourly data (8 Nordic zones).

Two inference paths:
  - method="mcmc": Nutpie/numpyro NUTS — proper posterior, slow at N>>100k.
  - method="svi":  ADVI mean-field variational — minutes instead of hours,
                   approximate posterior; good default when N is large.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

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


def _pick_nuts_sampler() -> str:
    """Prefer nutpie (fast, Rust-backed), fall back to numpyro (JAX), then stock PyMC."""
    try:
        import nutpie  # noqa: F401
        return "nutpie"
    except ImportError:
        pass
    try:
        import numpyro  # noqa: F401
        return "numpyro"
    except ImportError:
        logger.warning(
            "Neither nutpie nor numpyro installed; falling back to stock PyMC NUTS "
            "(order-of-magnitude slower). Install with: pip install nutpie"
        )
        return "pymc"


def _sample_mcmc(model, draws: int, tune: int, chains: int, random_seed: int):
    """Run NUTS sampling inside the given pm.Model context."""
    import pymc as pm
    sampler = _pick_nuts_sampler()
    logger.info("HSGP MCMC: sampler=%s  draws=%d  tune=%d  chains=%d", sampler, draws, tune, chains)
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=sampler,
            random_seed=random_seed,
        )


def _sample_svi(
    model,
    n_iter: int,
    n_samples: int,
    random_seed: int,
    method: str = "advi",
):
    """Fit ADVI (mean-field or full-rank) to the model and draw `n_samples`.

    method="advi" is mean-field (fast, assumes independent posterior factors).
    method="fullrank_advi" captures hyperparameter correlations — preferable
    for HSGP where ell_spatial, ell_temporal, eta are typically correlated,
    at the cost of O(K^2) variational parameters.

    Returns an InferenceData shaped like MCMC output (posterior group has
    dims (chain=1, draw=n_samples)), so downstream code — spatial_correlation,
    sample_posterior_predictive, diagnostics — works unchanged.
    """
    import pymc as pm
    if method not in ("advi", "fullrank_advi"):
        raise ValueError(f"SVI method must be 'advi' or 'fullrank_advi', got {method!r}")
    logger.info("HSGP SVI: method=%s  n_iter=%d  n_samples=%d", method, n_iter, n_samples)
    with model:
        approx = pm.fit(
            n=n_iter,
            method=method,
            random_seed=random_seed,
            progressbar=False,
        )
        # ELBO convergence signal: compare the last 5% of the trace to the
        # preceding 5%. If the improvement is still material, n_iter was too low
        # and the approximation is under-fit.
        try:
            hist = np.asarray(approx.hist)
            if hist.size >= 40:
                tail = int(max(1, 0.05 * hist.size))
                last = hist[-tail:].mean()
                prev = hist[-2 * tail:-tail].mean()
                rel_delta = abs(last - prev) / (abs(prev) + 1e-9)
                logger.info(
                    "ADVI ELBO: final=%.3g  last_5%%_mean=%.3g  prev_5%%_mean=%.3g  rel_delta=%.2e",
                    float(hist[-1]), float(last), float(prev), float(rel_delta),
                )
                if rel_delta > 1e-3:
                    logger.warning(
                        "ADVI ELBO still improving (rel_delta=%.2e > 1e-3) — "
                        "consider increasing svi_n_iter beyond %d for a tighter fit.",
                        float(rel_delta), n_iter,
                    )
        except Exception as e:
            logger.debug("ADVI ELBO convergence check skipped: %s", e)
        return approx.sample(n_samples, random_seed=random_seed)


def train_hsgp(
    grid_snapshots: pd.DataFrame,
    method: Literal["mcmc", "svi"] = "mcmc",
    m_spatial: int = DEFAULT_M_SPATIAL,
    m_temporal: int = DEFAULT_M_TEMPORAL,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    svi_method: Literal["advi", "fullrank_advi"] = "advi",
    svi_n_iter: int = 50_000,
    svi_n_samples: int = 2000,
    random_seed: int = 42,
    idata_path: str | None = None,
) -> dict:
    """
    Train HSGP via MCMC (NUTS) or SVI (ADVI).

    Args:
        grid_snapshots: DataFrame with columns timestamp, region_id, wind_speed_mps.
        method: "mcmc" for full NUTS (slow, accurate) or "svi" for ADVI
                (fast, approximate mean-field posterior). Use "svi" when N
                is large enough that MCMC wall-time is prohibitive.
        m_spatial, m_temporal: Number of HSGP basis functions per dimension.
        draws, tune, chains: MCMC params (ignored for SVI).
        svi_n_iter: ADVI optimisation steps (ignored for MCMC).
        svi_n_samples: Draws from the fitted ADVI approximation (ignored for MCMC).
        random_seed: Seed for sampler/optimiser.
        idata_path: If set, write InferenceData NetCDF here immediately after sampling.

    Returns:
        Dict with idata, model, gp, metadata, spatial_correlation, idata_path.
    """
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    import pymc as pm

    if method not in ("mcmc", "svi"):
        raise ValueError(f"method must be 'mcmc' or 'svi', got {method!r}")

    if grid_snapshots.empty or len(grid_snapshots) < 10:
        logger.warning("grid_snapshots empty or too small; returning stub result")
        return {"idata": None, "model": None, "gp": None, "metadata": {}, "spatial_correlation": None}

    X, y, metadata = prepare_hsgp_2d_input(grid_snapshots)
    n_obs = X.shape[0]
    coords = {"obs": np.arange(n_obs)}
    model, gp = build_hsgp_model(X, y, m_spatial=m_spatial, m_temporal=m_temporal, coords=coords)

    if method == "mcmc":
        idata = _sample_mcmc(model, draws=draws, tune=tune, chains=chains, random_seed=random_seed)
    else:
        idata = _sample_svi(
            model,
            n_iter=svi_n_iter,
            n_samples=svi_n_samples,
            random_seed=random_seed,
            method=svi_method,
        )

    # Save immediately after sampling to prevent data loss from late crashes
    if idata_path:
        idata.to_netcdf(idata_path)
        logger.info("Saved idata to %s", idata_path)

    # Both samplers can miss log_likelihood in the InferenceData — recompute
    # explicitly so downstream LOO/WAIC works. SVI produces only posterior,
    # so this adds the log_likelihood group in-place.
    try:
        pm.compute_log_likelihood(idata, model=model)
        if idata_path:
            idata.to_netcdf(idata_path)  # update with log_likelihood
    except Exception as e:
        logger.debug("log_likelihood computation skipped: %s", e)

    # Phase 1 diagnostics: divergences + r_hat + ESS (required before interpreting results)
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum().item())
        logger.info("Divergences: %d", n_div)
        if n_div > 0:
            logger.warning(
                "%d divergences detected — consider non-centered parameterization "
                "or increasing target_accept", n_div
            )

    spatial_correlation = None
    try:
        ell_s = idata.posterior["ell_spatial"].values.flatten()
        ell_t = idata.posterior["ell_temporal"].values.flatten()
        eta_post = idata.posterior["eta"].values.flatten()
        spatial_correlation = np.corrcoef(np.column_stack([ell_s, ell_t, eta_post])) if len(ell_s) > 1 else None
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
    import uuid
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

    # Use unique variable names so this function can be called multiple times
    # on the same model object (e.g. once per forecast horizon) without
    # triggering "Variable name already exists" errors.
    uid = uuid.uuid4().hex[:8]
    cond_name = f"fcond_{uid}"
    pred_name = f"y_new_{uid}"

    with model:
        fcond = gp.conditional(cond_name, Xnew=X_new)
        sigma = model["sigma"]
        y_new = pm.Normal(pred_name, mu=fcond, sigma=sigma, shape=(n_new,))
        pp = pm.sample_posterior_predictive(
            idata,
            var_names=[pred_name],
            predictions=True,
            random_seed=42,
        )
    samples = pp.predictions[pred_name].values
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


def compute_cvar(
    samples: np.ndarray,
    threshold_mps: float = 3.0,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Conditional Value-at-Risk: expected shortfall severity in the worst
    alpha-fraction of scenarios (Rockafellar-Uryasev formulation).

    While compute_tail_risk answers "what is the probability of wind drought?",
    CVaR answers "in the worst 5% of outcomes, how severe is the shortfall
    per zone?" — the information a grid operator needs for dispatch decisions.

    Args:
        samples: (n_samples, n_zones) wind speed draws from posterior predictive.
        threshold_mps: Wind speed below which generation is considered insufficient.
        alpha: Tail fraction (0.05 = worst 5% of scenarios).

    Returns:
        (n_zones,) expected shortfall (m/s below threshold) per zone,
        averaged over the worst-alpha scenarios. Zero means no shortfall
        in the tail.
    """
    # Per-sample, per-zone shortfall (clipped at zero — no "bonus" for high wind)
    shortfall = np.maximum(threshold_mps - samples, 0.0)  # (n_samples, n_zones)

    # Rank scenarios by total cross-zone shortfall (worst = highest total)
    total_shortfall = shortfall.sum(axis=1)  # (n_samples,)
    cutoff_idx = max(1, int(np.ceil(alpha * len(total_shortfall))))
    cutoff = np.sort(total_shortfall)[-cutoff_idx]

    tail_mask = total_shortfall >= cutoff
    if not tail_mask.any():
        return np.zeros(samples.shape[1])

    # Expected per-zone shortfall conditional on being in the tail
    return shortfall[tail_mask].mean(axis=0)


def compute_quantiles(
    samples: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict[str, np.ndarray]:
    """
    Compute wind speed quantiles from posterior predictive samples.

    Args:
        samples: (n_samples, n_zones) wind speed draws.
        quantiles: Tuple of quantile levels (e.g. 0.10 for P10).

    Returns:
        Dict mapping "p10", "p50", "p90" (etc.) to (n_zones,) arrays.
    """
    result = {}
    for q in quantiles:
        label = f"p{int(q * 100)}"
        result[label] = np.quantile(samples, q, axis=0)
    return result
