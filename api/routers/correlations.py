"""
Correlations router: /correlations
"""

from fastapi import APIRouter

from api.data import DEFAULT_ZONE_IDS, get_spatial_correlation

router = APIRouter()


@router.get("")
def get_correlations():
    """
    Get learned spatial correlation matrix between Nordic zones.
    Reads from MinIO (hsgp_model spatial_correlation) when available.
    """
    result = get_spatial_correlation()
    if result is not None:
        zone_ids, correlation_matrix = result
        return {
            "zone_ids": zone_ids,
            "correlation_matrix": correlation_matrix,
            "status": "ok",
        }
    return {
        "zone_ids": DEFAULT_ZONE_IDS,
        "correlation_matrix": [],
        "status": "stub",
    }
