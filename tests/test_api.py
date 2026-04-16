"""
Tests for the Grid Resilience FastAPI endpoints.

Uses TestClient (no real S3/MinIO needed). Data-layer functions are patched
so tests validate routing, validation, and response shape.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "grid-resilience-api"
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------

class TestForecast:
    def test_valid_region(self):
        with patch("api.routers.forecast.get_forecast_for_region", return_value=0.42):
            resp = client.get("/forecast/SE1?horizon_h=24")
        assert resp.status_code == 200
        body = resp.json()
        assert body["region_id"] == "SE1"
        assert body["p_shortfall"] == 0.42
        assert body["status"] == "ok"

    def test_case_insensitive_region(self):
        with patch("api.routers.forecast.get_forecast_for_region", return_value=0.1):
            resp = client.get("/forecast/se4?horizon_h=48")
        assert resp.status_code == 200
        assert resp.json()["region_id"] == "SE4"

    def test_invalid_region_returns_400(self):
        resp = client.get("/forecast/XX1?horizon_h=24")
        assert resp.status_code == 400
        assert "Invalid region_id" in resp.json()["detail"]

    def test_invalid_horizon_returns_400(self):
        resp = client.get("/forecast/SE1?horizon_h=12")
        assert resp.status_code == 400
        assert "Invalid horizon_h" in resp.json()["detail"]

    def test_stub_when_no_data(self):
        with patch("api.routers.forecast.get_forecast_for_region", return_value=None):
            resp = client.get("/forecast/NO1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stub"
        assert body["p_shortfall"] == 0.0


# ---------------------------------------------------------------------------
# /scenarios
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_list_scenarios(self):
        fake_data = [
            {"region_id": "SE1", "horizon_h": 24, "p_shortfall": 0.1,
             "cvar_shortfall": 0.5, "wind_p10": 2.1, "wind_p50": 7.0, "wind_p90": 12.3},
            {"region_id": "SE2", "horizon_h": 24, "p_shortfall": 0.2,
             "cvar_shortfall": 0.8, "wind_p10": 1.9, "wind_p50": 6.5, "wind_p90": 11.0},
        ]
        with patch("api.routers.scenarios.get_all_scenarios", return_value=fake_data):
            resp = client.get("/scenarios/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["count"] == 2
        assert len(body["scenarios"]) == 2

    def test_list_scenarios_unavailable(self):
        with patch("api.routers.scenarios.get_all_scenarios", return_value=None):
            resp = client.get("/scenarios/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unavailable"
        assert body["scenarios"] == []

    def test_single_scenario(self):
        fake = {
            "region_id": "DK1", "horizon_h": 48,
            "p_shortfall": 0.15, "cvar_shortfall": 0.7,
            "wind_p10": 2.5, "wind_p50": 6.8, "wind_p90": 11.5,
        }
        with patch("api.routers.scenarios.get_scenarios_for_region", return_value=fake):
            resp = client.get("/scenarios/DK1?horizon_h=48")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["cvar_shortfall"] == 0.7
        assert body["wind_p10"] == 2.5
        assert body["wind_p50"] == 6.8
        assert body["wind_p90"] == 11.5

    def test_single_scenario_unavailable(self):
        with patch("api.routers.scenarios.get_scenarios_for_region", return_value=None):
            resp = client.get("/scenarios/FI?horizon_h=72")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unavailable"
        assert body["p_shortfall"] is None
        assert body["cvar_shortfall"] is None

    def test_invalid_region_returns_400(self):
        resp = client.get("/scenarios/ZZ9")
        assert resp.status_code == 400

    def test_invalid_horizon_returns_400(self):
        resp = client.get("/scenarios/SE1?horizon_h=36")
        assert resp.status_code == 400
