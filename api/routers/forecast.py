"""
Forecast router: /forecast/{region_id}
"""

from fastapi import APIRouter, HTTPException

from api.data import get_forecast_for_region

router = APIRouter()

VALID_REGION_IDS = {"SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "DK1", "DK2", "FI"}
VALID_HORIZONS = {24, 48, 72}


@router.get("/{region_id}")
def get_forecast(region_id: str, horizon_h: int = 24):
    """
    Get tail-risk forecast for a region.

    Args:
        region_id: Zone ID (e.g. SE1, DK1).
        horizon_h: Forecast horizon in hours (24, 48, or 72).

    Returns:
        P(shortfall > threshold) for the given horizon. Reads from MinIO (tail_risk_forecasts) when available.
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
    p_shortfall = get_forecast_for_region(region_id, horizon_h)
    if p_shortfall is not None:
        return {
            "region_id": region_id,
            "horizon_h": horizon_h,
            "p_shortfall": p_shortfall,
            "status": "ok",
        }
    return {
        "region_id": region_id,
        "horizon_h": horizon_h,
        "p_shortfall": 0.0,
        "status": "stub",
    }
