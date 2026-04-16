"""
FastAPI forecast service for Grid Resilience.

Endpoints:
  GET /forecast/{region_id}    - Tail-risk forecast for a region (24/48/72h)
  GET /correlations            - Learned spatial correlation matrix
  GET /scenarios               - Full scenario matrix (all regions, all horizons)
  GET /scenarios/{region_id}   - Risk profile for one region (p_shortfall, CVaR, quantiles)
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI

from api.routers import forecast, correlations, scenarios

app = FastAPI(
    title="Grid Resilience API",
    description="Tail-risk forecasts, CVaR scenarios, and spatial correlations for Nordic grid zones",
    version="0.2.0",
)

app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(correlations.router, prefix="/correlations", tags=["correlations"])
app.include_router(scenarios.router, prefix="/scenarios", tags=["scenarios"])


@app.get("/")
def root():
    return {"service": "grid-resilience-api", "status": "ok"}
