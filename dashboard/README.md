# Grid Resilience Dashboard

The risk view (P10/P50/P90 shortfall by region, conditional risk) can be served by **Streamlit** (MVP) or **Grafana** (recommended if you already run Grafana in Kubernetes).

---

## Option A: Grafana (e.g. existing k8s Grafana)

Yes — you can use your existing Grafana pod. Grafana needs a **data source** that serves the pipeline outputs. Two straightforward options:

### 1. PostgreSQL data source (recommended)

- The pipeline already writes `tail_risk_forecasts` (and can write summary tables) to MinIO/Parquet. Add an optional step that **also** writes the same (or aggregated) data to **PostgreSQL** (e.g. a `risk_forecasts` or `dashboard_metrics` table).
- In Grafana, add a **PostgreSQL** data source pointing at the same Postgres you use for the pipeline (connection string from cluster secrets or env).
- Build dashboards: time series of `p_shortfall` by region, tables of P10/P50/P90 by zone, etc.

**Table shape example** (for Grafana):

| timestamp | region_id | horizon_h | p_shortfall |
|-----------|-----------|-----------|-------------|
| 2024-01-15 12:00 | SE4 | 24 | 0.82 |

### 2. HTTP/API data source

- Expose the **FastAPI** forecast service (e.g. `/forecast/{region_id}`, `/correlations`) inside the cluster.
- In Grafana, use the **Infinity** or **JSON API** data source to query that API and parse the response.
- Dashboards can show tables or time series from the API JSON.

**Summary:** Use **PostgreSQL** if you want simple, robust dashboards and can write pipeline results to Postgres. Use **HTTP/API** if you prefer not to duplicate data and your FastAPI is reachable from Grafana.

---

## Option B: Streamlit (MVP)

Streamlit app showing the Risk Map: P10/P50/P90 shortfall by region, conditional risk (e.g., SE4 high if DE_Wind < 10%). Reads from MinIO Gold (inference artifacts) or the FastAPI forecast service.

**Port:** 8501

### Run

See "How to Run" in the root [README](../README.md). For Kubernetes, port-forward the Streamlit service: `kubectl port-forward svc/grid-dashboard 8501:8501`.
