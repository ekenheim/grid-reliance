"""
Forecast router: /forecast/{region_id}
"""

from fastapi import APIRouter

from api.data import get_forecast_for_region

router = APIRouter()


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
