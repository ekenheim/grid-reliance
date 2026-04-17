#!/usr/bin/env python3
"""
Full-platform inference driver with Ray-distributed MCMC chain sampling.

Each MCMC chain runs as an independent ray.remote task on a separate Ray worker,
enabling parallel sampling across the ray-kuberay cluster. InferenceData objects
are serialised to NetCDF bytes for transport and concatenated on the driver.

Entrypoint for the RayJob (bayesian-grid-model.yaml):
  python -m inference.run_inference

Env (Gold = write bucket; Silver = read bucket):
  Gold:   BUCKET_NAME, BUCKET_HOST, BUCKET_PORT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  Silver: SILVER_BUCKET_NAME, SILVER_BUCKET_HOST, SILVER_BUCKET_PORT,
          SILVER_AWS_ACCESS_KEY_ID, SILVER_AWS_SECRET_ACCESS_KEY

Silver object key: SILVER_PREFIX/grid_snapshots.parquet (default silver/grid_snapshots.parquet)
  DataFrame must have columns: timestamp, region_id, wind_speed_mps
"""

from __future__ import annotations

import io
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
import boto3
from botocore.config import Config

DEFAULT_ZONE_IDS = ["SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"]
PREFIX = "dagster"
FORECASTS_KEY = f"{PREFIX}/tail_risk_forecasts.parquet"
MODEL_KEY = f"{PREFIX}/hsgp_model.pkl"
TEMPORAL_SCALE_H = 168.0

# Number of chains to distribute across Ray workers.
# Each chain runs on one worker with num_cpus=CPUS_PER_CHAIN and up to 6 GiB RAM.
# HSGP gradient evaluation is BLAS-bound at N>>M, so bumping CPUS_PER_CHAIN above
# 1 gives real per-chain speedups via multi-threaded matmul (OMP/MKL). Returns
# saturate around 4 cores because the N×M design matrix exceeds L3 cache and
# memory bandwidth becomes the bottleneck.
N_CHAINS = int(os.environ.get("HSGP_N_CHAINS", "4"))
CPUS_PER_CHAIN = int(os.environ.get("HSGP_CPUS_PER_CHAIN", "4"))
MEMORY_PER_CHAIN_GIB = int(os.environ.get("HSGP_MEMORY_PER_CHAIN_GIB", "6"))

# Inference method: "mcmc" (chain-parallel NUTS) or "svi" (single-task ADVI).
# SVI runs on one worker and typically finishes in minutes vs. hours for MCMC
# at N>>100k. Trades exact posterior for a mean-field approximation.
HSGP_METHOD = os.environ.get("HSGP_METHOD", "mcmc").lower()

DRAWS = int(os.environ.get("HSGP_DRAWS", "300"))
TUNE = int(os.environ.get("HSGP_TUNE", "200"))
SVI_METHOD = os.environ.get("HSGP_SVI_METHOD", "advi").lower()
SVI_N_ITER = int(os.environ.get("HSGP_SVI_N_ITER", "50000"))
SVI_N_SAMPLES = int(os.environ.get("HSGP_SVI_N_SAMPLES", "2000"))
M_SPATIAL = 5
M_TEMPORAL = 10


# ---------------------------------------------------------------------------
# S3 helpers (identical to the previous single-process version)
# ---------------------------------------------------------------------------

def _s3_client_from_env(prefix: str = ""):
    """Build (client, bucket) from env. prefix='' for Gold, 'SILVER_' for Silver."""
    if prefix:
        endpoint = os.environ.get(f"{prefix}S3_ENDPOINT", "").strip()
        if not endpoint:
            host = (os.environ.get(f"{prefix}BUCKET_HOST") or "").strip()
            port = (os.environ.get(f"{prefix}BUCKET_PORT") or "").strip()
            endpoint = f"http://{host}:{port}" if host and port else (f"http://{host}" if host else "")
        access = os.environ.get(f"{prefix}AWS_ACCESS_KEY_ID") or os.environ.get(f"{prefix}MINIO_ACCESS_KEY", "")
        secret = os.environ.get(f"{prefix}AWS_SECRET_ACCESS_KEY") or os.environ.get(f"{prefix}MINIO_SECRET_KEY", "")
        bucket = os.environ.get(f"{prefix}BUCKET_NAME") or os.environ.get(f"{prefix}MINIO_BUCKET", "")
    else:
        endpoint = (os.environ.get("MINIO_ENDPOINT") or os.environ.get("S3_ENDPOINT") or "").strip()
        if not endpoint:
            host = (os.environ.get("BUCKET_HOST") or "").strip()
            port = (os.environ.get("BUCKET_PORT") or "").strip()
            endpoint = f"http://{host}:{port}" if host and port else (f"http://{host}" if host else "")
        access = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("MINIO_ACCESS_KEY", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("MINIO_SECRET_KEY", "")
        bucket = os.environ.get("BUCKET_NAME") or os.environ.get("MINIO_BUCKET", "grid-resilience-gold")
    if not endpoint or not access or not secret:
        return None, None
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    region = (
        os.environ.get(f"{prefix}BUCKET_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return client, bucket or None


def load_grid_snapshots_from_silver() -> pd.DataFrame:
    client, bucket = _s3_client_from_env("SILVER_")
    if not client or not bucket:
        raise RuntimeError(
            "Silver not configured: set SILVER_BUCKET_NAME, SILVER_BUCKET_HOST, "
            "SILVER_BUCKET_PORT, SILVER_AWS_ACCESS_KEY_ID, SILVER_AWS_SECRET_ACCESS_KEY"
        )
    key = os.environ.get("SILVER_PREFIX", "silver").rstrip("/") + "/grid_snapshots.parquet"
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as e:
        raise RuntimeError(f"Failed to load s3://{bucket}/{key}: {e}") from e


def write_forecasts_and_model_to_gold(forecasts_df: pd.DataFrame, model_dict: dict) -> None:
    client, bucket = _s3_client_from_env("")
    if not client or not bucket:
        raise RuntimeError(
            "Gold not configured: set BUCKET_NAME, BUCKET_HOST, BUCKET_PORT, "
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )
    buf = io.BytesIO()
    forecasts_df.to_parquet(buf, index=False)
    buf.seek(0)
    client.put_object(Bucket=bucket, Key=FORECASTS_KEY, Body=buf.getvalue())

    buf_pkl = io.BytesIO()
    pickle.dump(model_dict, buf_pkl)
    buf_pkl.seek(0)
    client.put_object(Bucket=bucket, Key=MODEL_KEY, Body=buf_pkl.getvalue())


# ---------------------------------------------------------------------------
# Ray-distributed chain sampling
# ---------------------------------------------------------------------------

def _set_thread_env(n_cpus: int) -> None:
    """Pin BLAS/XLA thread counts to the Ray reservation so we use the CPUs we
    asked for (and no more — otherwise multiple chains on one node oversubscribe
    and slow each other down). Must be called before importing numpy/jax/pymc."""
    n = str(n_cpus)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, n)
    # JAX/XLA: pin intra-op parallelism; inter-op stays at 1 because a single
    # MCMC chain has no inter-op parallelism to exploit.
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_cpus}",
    )


def _sample_one_chain(
    X: np.ndarray,
    y: np.ndarray,
    chain_idx: int,
    m_spatial: int,
    m_temporal: int,
    draws: int,
    tune: int,
    random_seed: int,
    cpus_per_chain: int = 2,
) -> bytes:
    """
    Run a single MCMC chain and return the InferenceData serialised as NetCDF bytes.

    Designed to run as a ray.remote task. Each call to pm.sample uses chains=1
    so the full chain count is achieved by launching N parallel tasks. The
    random_seed is offset by chain_idx so each chain explores independently.
    """
    _set_thread_env(cpus_per_chain)
    import pymc as pm
    from pipeline.model.hsgp_model import build_hsgp_model

    n_obs = X.shape[0]
    coords = {"obs": np.arange(n_obs)}
    model, _ = build_hsgp_model(X, y, m_spatial=m_spatial, m_temporal=m_temporal, coords=coords)

    try:
        import nutpie  # noqa: F401
        nuts_sampler = "nutpie"
    except ImportError:
        try:
            import numpyro  # noqa: F401
            nuts_sampler = "numpyro"
        except ImportError:
            nuts_sampler = "pymc"

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed + chain_idx,
            progressbar=False,
        )

    buf = io.BytesIO()
    idata.to_netcdf(buf)
    return buf.getvalue()


def _fit_svi(
    X: np.ndarray,
    y: np.ndarray,
    m_spatial: int,
    m_temporal: int,
    n_iter: int,
    n_samples: int,
    random_seed: int,
    svi_method: str = "advi",
    cpus: int = 4,
) -> bytes:
    """
    Fit ADVI on one worker and return InferenceData bytes. SVI is not
    chain-parallelisable (it's a single optimisation), so distributing across
    Ray makes no sense — we run it on exactly one worker with more CPUs.
    """
    _set_thread_env(cpus)
    import pymc as pm
    from pipeline.model.hsgp_model import build_hsgp_model

    n_obs = X.shape[0]
    coords = {"obs": np.arange(n_obs)}
    model, _ = build_hsgp_model(X, y, m_spatial=m_spatial, m_temporal=m_temporal, coords=coords)
    with model:
        approx = pm.fit(n=n_iter, method=svi_method, random_seed=random_seed, progressbar=False)
        idata = approx.sample(n_samples, random_seed=random_seed)
    buf = io.BytesIO()
    idata.to_netcdf(buf)
    return buf.getvalue()


def train_hsgp_distributed(
    grid_snapshots: pd.DataFrame,
    method: str = "mcmc",
    m_spatial: int = M_SPATIAL,
    m_temporal: int = M_TEMPORAL,
    draws: int = DRAWS,
    tune: int = TUNE,
    n_chains: int = N_CHAINS,
    cpus_per_chain: int = CPUS_PER_CHAIN,
    memory_per_chain_gib: int = MEMORY_PER_CHAIN_GIB,
    svi_method: str = SVI_METHOD,
    svi_n_iter: int = SVI_N_ITER,
    svi_n_samples: int = SVI_N_SAMPLES,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Train the HSGP model on the Ray cluster.

    method="mcmc": chain sampling distributed across workers (one chain per
    ray.remote task, num_cpus=cpus_per_chain each). InferenceData from all
    chains is concatenated with arviz.concat.

    method="svi": single ADVI task on one worker. SVI is a single optimisation,
    not chain-parallelisable — it gets one worker with more CPUs.

    Returns the same dict shape as pipeline.model.hsgp_model.train_hsgp.
    """
    import ray
    import arviz as az
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    from pipeline.model.hsgp_model import build_hsgp_model

    X, y, metadata = prepare_hsgp_2d_input(grid_snapshots)
    n_obs = X.shape[0]

    if method == "svi":
        print(
            f"Starting Ray SVI task  |  n_obs={n_obs}  m=[{m_spatial},{m_temporal}]  "
            f"n_iter={svi_n_iter}  n_samples={svi_n_samples}  cpus={cpus_per_chain}",
            file=sys.stderr,
        )
        svi_remote = ray.remote(
            num_cpus=cpus_per_chain,
            memory=memory_per_chain_gib * 1024 ** 3,
        )(_fit_svi)
        svi_bytes = ray.get(svi_remote.remote(
            X=X, y=y,
            m_spatial=m_spatial, m_temporal=m_temporal,
            n_iter=svi_n_iter, n_samples=svi_n_samples,
            random_seed=random_seed,
            svi_method=svi_method,
            cpus=cpus_per_chain,
        ))
        idata = az.from_netcdf(io.BytesIO(svi_bytes))
        print("SVI fit complete.", file=sys.stderr)
    else:
        print(
            f"Starting {n_chains} Ray chains  |  n_obs={n_obs}  "
            f"m=[{m_spatial},{m_temporal}]  draws={draws}  tune={tune}  "
            f"cpus_per_chain={cpus_per_chain}",
            file=sys.stderr,
        )
        # PyMC/nutpie/numpyro are pre-installed in the grid-resilience-ray-worker
        # image — no runtime_env pip install needed.
        sample_remote = ray.remote(
            num_cpus=cpus_per_chain,
            memory=memory_per_chain_gib * 1024 ** 3,
        )(_sample_one_chain)

        futures = [
            sample_remote.remote(
                X=X, y=y,
                chain_idx=i,
                m_spatial=m_spatial,
                m_temporal=m_temporal,
                draws=draws,
                tune=tune,
                random_seed=random_seed,
                cpus_per_chain=cpus_per_chain,
            )
            for i in range(n_chains)
        ]

        chain_bytes: list[bytes] = ray.get(futures)
        print(f"All {n_chains} chains complete.", file=sys.stderr)

        # Deserialise and concatenate across the "chain" dimension
        idatas = [az.from_netcdf(io.BytesIO(b)) for b in chain_bytes]
        idata = az.concat(idatas, dim="chain")

    # Derive spatial correlation proxy from posterior hyperparameters
    spatial_correlation: np.ndarray | None = None
    try:
        ell_s = idata.posterior["ell_spatial"].values.flatten()
        ell_t = idata.posterior["ell_temporal"].values.flatten()
        eta = idata.posterior["eta"].values.flatten()
        spatial_correlation = (
            np.corrcoef(np.column_stack([ell_s, ell_t, eta])) if len(ell_s) > 1 else None
        )
    except Exception:
        pass

    # Rebuild model object (needed for sample_posterior_predictive)
    coords = {"obs": np.arange(n_obs)}
    model, gp = build_hsgp_model(X, y, m_spatial=m_spatial, m_temporal=m_temporal, coords=coords)

    return {
        "idata": idata,
        "model": model,
        "gp": gp,
        "metadata": metadata,
        "spatial_correlation": spatial_correlation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import ray
    from pipeline.model.feature_engineering import prepare_hsgp_2d_input
    from pipeline.model.hsgp_model import (
        build_hsgp_model,
        compute_tail_risk,
        compute_cvar,
        compute_quantiles,
        sample_posterior_predictive,
    )

    # Connect to the existing ray-kuberay cluster
    ray.init(address="auto")
    print(f"Ray cluster resources: {ray.cluster_resources()}", file=sys.stderr)

    print("Loading grid_snapshots from Silver...", file=sys.stderr)
    grid_snapshots = load_grid_snapshots_from_silver()
    if grid_snapshots.empty or len(grid_snapshots) < 10:
        print("grid_snapshots empty or too small; exiting.", file=sys.stderr)
        return 1

    if "wind_speed_mps" not in grid_snapshots.columns and "wind_speed" in grid_snapshots.columns:
        grid_snapshots = grid_snapshots.rename(columns={"wind_speed": "wind_speed_mps"})

    print(
        f"Loaded {len(grid_snapshots):,} rows  |  "
        f"{grid_snapshots['region_id'].nunique()} zones  |  "
        f"{grid_snapshots['timestamp'].nunique()} timesteps",
        file=sys.stderr,
    )

    result = train_hsgp_distributed(
        grid_snapshots,
        method=HSGP_METHOD,
        m_spatial=M_SPATIAL,
        m_temporal=M_TEMPORAL,
        draws=DRAWS,
        tune=TUNE,
        n_chains=N_CHAINS,
        cpus_per_chain=CPUS_PER_CHAIN,
        memory_per_chain_gib=MEMORY_PER_CHAIN_GIB,
        svi_method=SVI_METHOD,
        svi_n_iter=SVI_N_ITER,
        svi_n_samples=SVI_N_SAMPLES,
        random_seed=42,
    )

    if result.get("idata") is None:
        print("No idata returned; exiting.", file=sys.stderr)
        return 1

    zone_ids = result.get("metadata", {}).get("zone_ids", DEFAULT_ZONE_IDS)
    if not zone_ids and "region_id" in grid_snapshots.columns:
        zone_ids = sorted(grid_snapshots["region_id"].unique().tolist())
    if not zone_ids:
        zone_ids = list(DEFAULT_ZONE_IDS)

    X_full, y_full, _ = prepare_hsgp_2d_input(grid_snapshots, temporal_scale_h=TEMPORAL_SCALE_H)
    model, gp = build_hsgp_model(X_full, y_full, m_spatial=M_SPATIAL, m_temporal=M_TEMPORAL)
    result_tmp = {"idata": result["idata"], "model": model, "gp": gp}

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
        spatial_coords_new = np.array(
            [zone_to_idx[z] / max(n_zones - 1, 1) for z in zone_ids], dtype=np.float64
        )
        temporal_coords_new = np.full(n_zones, temporal_norm, dtype=np.float64)
        samples = sample_posterior_predictive(
            result_tmp, spatial_coords_new, temporal_coords_new, n_samples=500
        )
        p_shortfall = compute_tail_risk(samples, threshold_mps=3.0)
        cvar = compute_cvar(samples, threshold_mps=3.0, alpha=0.05)
        quantiles = compute_quantiles(samples, quantiles=(0.10, 0.50, 0.90))
        for i, zid in enumerate(zone_ids):
            rows.append({
                "timestamp": t_new,
                "region_id": zid,
                "horizon_h": horizon_h,
                "p_shortfall": float(p_shortfall[i]),
                "cvar_shortfall": float(cvar[i]),
                "wind_p10": float(quantiles["p10"][i]),
                "wind_p50": float(quantiles["p50"][i]),
                "wind_p90": float(quantiles["p90"][i]),
            })

    forecasts_df = pd.DataFrame(rows)
    model_dict = {
        "metadata": result.get("metadata", {}),
        "spatial_correlation": result.get("spatial_correlation"),
        "m_spatial": M_SPATIAL,
        "m_temporal": M_TEMPORAL,
        "zone_ids": zone_ids,
    }

    print("Writing forecasts and model artefacts to Gold...", file=sys.stderr)
    write_forecasts_and_model_to_gold(forecasts_df, model_dict)
    print(f"Wrote {FORECASTS_KEY} and {MODEL_KEY} to Gold.", file=sys.stderr)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
