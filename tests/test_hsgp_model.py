"""
Tests for the HSGP model (pipeline.model.hsgp_model).

Uses pymc.testing.mock_sample so pm.sample() returns prior predictive draws
instead of running real MCMC — tests run in seconds and validate model
structure, not posterior values.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pipeline.model.hsgp_model import (
    build_hsgp_model,
    compute_tail_risk,
    train_hsgp,
)
from pipeline.model.feature_engineering import (
    prepare_hsgp_2d_input,
    ZONE_CENTROIDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_grid_snapshots() -> pd.DataFrame:
    """Small synthetic DataFrame matching the pipeline's grid_snapshots schema."""
    rng = np.random.default_rng(42)
    zones = ["SE1", "SE2", "SE3", "SE4"]
    timestamps = pd.date_range("2023-01-01", periods=48, freq="h")
    rows = []
    for ts in timestamps:
        for z in zones:
            rows.append({
                "timestamp": ts,
                "region_id": z,
                "wind_speed_mps": float(rng.normal(7.0, 2.0)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def hsgp_Xy(synthetic_grid_snapshots):
    """Pre-computed X, y arrays for build_hsgp_model."""
    X, y, metadata = prepare_hsgp_2d_input(synthetic_grid_snapshots)
    return X, y, metadata


# ---------------------------------------------------------------------------
# Model structure tests (mock sampling)
# ---------------------------------------------------------------------------

class TestBuildHSGPModel:
    def test_returns_model_and_gp(self, hsgp_Xy):
        X, y, _ = hsgp_Xy
        model, gp = build_hsgp_model(X, y)
        assert isinstance(model, pm.Model)
        assert gp is not None

    def test_model_has_expected_free_variables(self, hsgp_Xy):
        X, y, _ = hsgp_Xy
        model, _ = build_hsgp_model(X, y)
        free_rv_names = {rv.name for rv in model.free_RVs}
        assert "ell_spatial" in free_rv_names
        assert "ell_temporal" in free_rv_names
        assert "eta" in free_rv_names
        assert "sigma" in free_rv_names

    def test_model_has_observed_variable(self, hsgp_Xy):
        X, y, _ = hsgp_Xy
        model, _ = build_hsgp_model(X, y)
        obs_names = {rv.name for rv in model.observed_RVs}
        assert "y_obs" in obs_names

    def test_custom_basis_functions(self, hsgp_Xy):
        X, y, _ = hsgp_Xy
        model, _ = build_hsgp_model(X, y, m_spatial=3, m_temporal=5)
        assert isinstance(model, pm.Model)

    def test_model_logp_evaluates(self, hsgp_Xy):
        """model.point_logps() should return finite values at the initial point."""
        X, y, _ = hsgp_Xy
        model, _ = build_hsgp_model(X, y)
        logps = model.point_logps()
        for name, val in logps.items():
            assert np.isfinite(val), f"{name} has non-finite logp: {val}"


class TestTrainHSGP:
    def test_train_returns_expected_keys(self, mock_pymc_sample, synthetic_grid_snapshots):
        result = train_hsgp(
            synthetic_grid_snapshots,
            m_spatial=3,
            m_temporal=5,
            draws=10,
            tune=10,
            chains=2,
        )
        assert "idata" in result
        assert "model" in result
        assert "gp" in result
        assert "metadata" in result
        assert "spatial_correlation" in result

    def test_train_returns_idata_with_posterior(self, mock_pymc_sample, synthetic_grid_snapshots):
        result = train_hsgp(
            synthetic_grid_snapshots,
            draws=10,
            tune=10,
            chains=2,
        )
        idata = result["idata"]
        assert idata is not None
        assert hasattr(idata, "posterior")
        assert "ell_spatial" in idata.posterior
        assert "ell_temporal" in idata.posterior
        assert "eta" in idata.posterior
        assert "sigma" in idata.posterior

    def test_train_empty_df_returns_stub(self, mock_pymc_sample):
        empty_df = pd.DataFrame(columns=["timestamp", "region_id", "wind_speed_mps"])
        result = train_hsgp(empty_df)
        assert result["idata"] is None
        assert result["model"] is None

    def test_train_metadata_has_zone_ids(self, mock_pymc_sample, synthetic_grid_snapshots):
        result = train_hsgp(
            synthetic_grid_snapshots,
            draws=10,
            tune=10,
            chains=2,
        )
        metadata = result["metadata"]
        assert "zone_ids" in metadata
        assert set(metadata["zone_ids"]) == {"SE1", "SE2", "SE3", "SE4"}


# ---------------------------------------------------------------------------
# Feature engineering tests (no sampling needed)
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_prepare_hsgp_2d_input_shape(self, synthetic_grid_snapshots):
        X, y, metadata = prepare_hsgp_2d_input(synthetic_grid_snapshots)
        n_expected = len(synthetic_grid_snapshots)
        assert X.shape == (n_expected, 2)
        assert y.shape == (n_expected,)

    def test_spatial_coords_normalized(self, synthetic_grid_snapshots):
        X, _, _ = prepare_hsgp_2d_input(synthetic_grid_snapshots)
        spatial = X[:, 0]
        assert spatial.min() >= 0.0
        assert spatial.max() <= 1.0

    def test_temporal_coords_positive(self, synthetic_grid_snapshots):
        X, _, _ = prepare_hsgp_2d_input(synthetic_grid_snapshots)
        temporal = X[:, 1]
        assert temporal.min() >= 0.0

    def test_zone_centroids_complete(self):
        expected_zones = {"SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"}
        assert set(ZONE_CENTROIDS.keys()) == expected_zones


# ---------------------------------------------------------------------------
# Tail risk computation tests (pure numpy, no sampling)
# ---------------------------------------------------------------------------

class TestComputeTailRisk:
    def test_all_below_threshold(self):
        samples = np.full((100, 3), 1.0)  # all below 3.0
        p = compute_tail_risk(samples, threshold_mps=3.0)
        np.testing.assert_array_equal(p, [1.0, 1.0, 1.0])

    def test_all_above_threshold(self):
        samples = np.full((100, 3), 10.0)  # all above 3.0
        p = compute_tail_risk(samples, threshold_mps=3.0)
        np.testing.assert_array_equal(p, [0.0, 0.0, 0.0])

    def test_half_below(self):
        samples = np.vstack([
            np.full((50, 2), 1.0),
            np.full((50, 2), 10.0),
        ])
        p = compute_tail_risk(samples, threshold_mps=3.0)
        np.testing.assert_allclose(p, [0.5, 0.5])

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(7, 2, size=(200, 5))
        p = compute_tail_risk(samples, threshold_mps=3.0)
        assert p.shape == (5,)
        assert np.all((p >= 0) & (p <= 1))
