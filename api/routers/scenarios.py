"""
Scenarios router: /scenarios

Exposes the full probabilistic risk profile (p_shortfall, CVaR, P10/P50/P90
wind speed quantiles) for downstream optimization consumers.
"""

from fastapi import APIRouter, HTTPException

from api.data import get_scenarios_for_region, get_all_scenarios

router = APIRouter()

VALID_REGION_IDS = {"SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"}
VALID_HORIZONS = {24, 48, 72}


@router.get("/")
def list_scenarios():
    """Return the full scenario matrix: all regions, all horizons.

    Each entry includes p_shortfall, cvar_shortfall, and wind speed quantiles
    (P10/P50/P90) from the posterior predictive distribution.
    """
    scenarios = get_all_scenarios()
    if scenarios is None:
        return {"scenarios": [], "status": "unavailable"}
    return {"scenarios": scenarios, "status": "ok", "count": len(scenarios)}


@router.get("/{region_id}")
def get_scenario(region_id: str, horizon_h: int = 24):
    """Return the risk profile for a single region and horizon.

    Response fields:
      - p_shortfall: P(wind < 3 m/s)
      - cvar_shortfall: expected shortfall severity in worst 5% of scenarios (m/s)
      - wind_p10/p50/p90: wind speed quantiles from posterior predictive
    """
    region_id = region_id.upper()
    if region_id not in VALID_REGION_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region_id '{region_id}'. Must be one of: {sorted(VALID_REGION_IDS)}",
        )
    if horizon_h not in VALID_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon_h {horizon_h}. Must be one of: {sorted(VALID_HORIZONS)}",
        )
    scenario = get_scenarios_for_region(region_id, horizon_h)
    if scenario is not None:
        return {**scenario, "status": "ok"}
    return {
        "region_id": region_id,
        "horizon_h": horizon_h,
        "p_shortfall": None,
        "cvar_shortfall": None,
        "wind_p10": None,
        "wind_p50": None,
        "wind_p90": None,
        "status": "unavailable",
    }
