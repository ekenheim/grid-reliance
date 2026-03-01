"""
MLflow utilities for logging HSGP model runs.

Shared by the Dagster pipeline and the model development notebook.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_mlflow_tracking_uri() -> str:
    """Get MLflow tracking URI from env or default."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def log_hsgp_run(
    trace_or_guide: Any,
    diagnostics: dict,
    metadata: dict,
    run_name: str | None = None,
    *,
    tracking_uri: str | None = None,
    log_artifacts: bool = True,
    m_spatial: int = 5,
    m_temporal: int = 10,
    n_iter: int = 20_000,
    spatial_correlation: np.ndarray | None = None,
    idata_path: str | None = None,
) -> None:
    """
    Log an HSGP model run to MLflow.

    Parameters
    ----------
    trace_or_guide : SVI result or guide (or None for MCMC-only).
        Trained model output.
    diagnostics : dict
        Output from run_diagnostics() (elbo_final, crps, n_divergences, max_r_hat, min_ess_bulk, etc.).
    metadata : dict
        Zone IDs, n_zones, n_timesteps.
    run_name : str, optional
        Custom run name.
    tracking_uri : str, optional
        MLflow server URI.
    log_artifacts : bool
        If True, upload spatial correlation matrix and metadata.
    m_spatial, m_temporal, n_iter : int
        Model/sampling parameters.
    spatial_correlation : ndarray, optional
        Learned spatial correlation matrix to log as artifact.
    idata_path : str, optional
        Path to saved InferenceData NetCDF (for MCMC runs).
    """
    import mlflow

    uri = tracking_uri or get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("grid-resilience-hsgp")

    name = run_name or "hsgp_dagster"

    with mlflow.start_run(run_name=name):
        mlflow.log_param("m_spatial", m_spatial)
        mlflow.log_param("m_temporal", m_temporal)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("n_zones", metadata.get("n_zones", 0))

        for k, v in diagnostics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        mlflow.set_tag("model_type", "hsgp")
        mlflow.set_tag("kernel", "matern_separable")
        if idata_path:
            mlflow.log_param("idata_path", idata_path)

        if log_artifacts and spatial_correlation is not None:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                np.save(f.name, spatial_correlation)
                f.close()
                try:
                    mlflow.log_artifact(f.name, artifact_path="spatial_correlation")
                finally:
                    try:
                        os.unlink(f.name)
                    except OSError:
                        pass

        if log_artifacts and metadata:
            meta_serializable = {k: v for k, v in metadata.items() if not callable(v)}
            if "zone_ids" in meta_serializable:
                meta_serializable["zone_ids"] = list(meta_serializable["zone_ids"])
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump(meta_serializable, f, indent=2)
                f.flush()
                f.close()
                try:
                    mlflow.log_artifact(f.name, artifact_path="metadata")
                finally:
                    try:
                        os.unlink(f.name)
                    except OSError:
                        pass

        logger.info("Logged HSGP run to MLflow: %s", name)
