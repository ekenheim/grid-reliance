"""
Model diagnostics for the HSGP pipeline.

Provides ELBO convergence, CRPS, spatial correlation validation, and MCMC diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_mcmc_diagnostics(
    idata: Any,
    var_names: list[str] | None = None,
    exclude_aux: bool = True,
) -> dict:
    """
    Run MCMC diagnostics on InferenceData (divergences, r_hat, ESS).

    Returns:
        Dict with n_divergences, max_r_hat, min_ess_bulk, min_ess_tail, summary (optional).
    """
    import arviz as az

    results = {}
    if idata is None:
        return results
    try:
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            n_div = int(idata.sample_stats["diverging"].sum().item())
            results["n_divergences"] = n_div
            logger.info("Divergences: %d", n_div)
        summary = az.summary(
            idata,
            var_names=var_names,
            filter_vars="all" if not exclude_aux else "like",
        )
        if summary is not None and len(summary) > 0:
            if "r_hat" in summary.columns:
                results["max_r_hat"] = float(summary["r_hat"].max())
            if "ess_bulk" in summary.columns:
                results["min_ess_bulk"] = float(summary["ess_bulk"].min())
            if "ess_tail" in summary.columns:
                results["min_ess_tail"] = float(summary["ess_tail"].min())
            logger.info(
                "r_hat max=%.4f, ess_bulk min=%.0f, ess_tail min=%.0f",
                results.get("max_r_hat", 0),
                results.get("min_ess_bulk", 0),
                results.get("min_ess_tail", 0),
            )
    except Exception as e:
        logger.warning("MCMC diagnostics failed: %s", e)
    return results


def run_diagnostics(
    trace_or_guide: Any,
    elbo_history: list[float] | None = None,
    spatial_correlation: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    posterior_predictive: np.ndarray | None = None,
    idata: Any = None,
) -> dict:
    """
    Run diagnostics on HSGP result (SVI or MCMC).

    If idata is provided, runs MCMC diagnostics (divergences, r_hat, ESS) and merges.

    Returns:
        Dict with elbo_final, elbo_converged, crps (if observed/pp provided),
        spatial_corr_norm (if spatial_correlation provided),
        and MCMC metrics if idata provided.
    """
    results = {}
    if idata is not None:
        results.update(run_mcmc_diagnostics(idata))

    if elbo_history is not None:
        results["elbo_final"] = float(elbo_history[-1]) if elbo_history else 0.0
        results["elbo_converged"] = _check_elbo_convergence(elbo_history)
        logger.info("ELBO final: %.2f, converged: %s", results["elbo_final"], results["elbo_converged"])

    if spatial_correlation is not None:
        # Frobenius norm of correlation matrix (sanity check)
        results["spatial_corr_norm"] = float(np.linalg.norm(spatial_correlation, "fro"))
        logger.info("Spatial correlation Frobenius norm: %.4f", results["spatial_corr_norm"])

    if observed is not None and posterior_predictive is not None:
        crps = _compute_crps(observed, posterior_predictive)
        results["crps"] = float(crps)
        logger.info("CRPS: %.4f", results["crps"])

    return results


def _check_elbo_convergence(elbo_history: list[float], window: int = 500) -> bool:
    """Check if ELBO has stabilized (monotonic decrease with noise)."""
    if len(elbo_history) < 2 * window:
        return False
    recent = elbo_history[-window:]
    return np.std(recent) < abs(np.mean(recent)) * 0.01


def _compute_crps(observed: np.ndarray, samples: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (lower is better).

    samples: (n_samples, n_obs)
    observed: (n_obs,)
    """
    # Simplified CRPS: mean absolute error of median forecast
    median_forecast = np.median(samples, axis=0)
    return float(np.mean(np.abs(observed - median_forecast)))
