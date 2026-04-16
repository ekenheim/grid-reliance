# Synthetic Data Generator

MVP path: synthetic data for local development. For production, the platform ingests real ERA5 + ENTSO-E data via `data-engineering/`.

Generates synthetic weather and grid data for 8 Nordic zones × 5 years hourly.

## Features

- **Matern spatial covariance**: Uses great-circle distances between zone centroids to generate correlated wind speeds.
- **Correlated drought events**: 5-10 events per year where multiple regions simultaneously experience low wind.
- **Price spikes**: 20-30 events (>200 EUR/MWh) that correlate with wind shortfalls.
- **PostgreSQL output**: Partitioned tables (weather_obs, grid_demand, spot_prices).
- **Optional MinIO**: Can write Parquet to MinIO for pipeline ingestion.

## Container image

The image runs as **UID 1000** (user `generator`). Platform manifests should set:

```yaml
securityContext:
  runAsUser: 1000
  runAsNonRoot: true
  fsGroup: 1000
```

## Usage

**Start PostgreSQL (from repo root):**

```bash
docker compose up -d postgres
```

**Run generator:**

```bash
pip install -r requirements.txt
# Option A: Copy .env.example to .env (from repo root) — no manual export needed
# Option B: Set env var manually:
#   Bash: export DATABASE_URL=postgresql://grid_resilience:grid_resilience@localhost:5432/grid_resilience
#   PowerShell: $env:DATABASE_URL = "postgresql://grid_resilience:grid_resilience@localhost:5432/grid_resilience"
python generate.py
```

## Zone list (8 Nordic zones)

- SE1, SE2, SE3, SE4 (Sweden)
- NO1, NO2 (Norway)
- DK1, DK2 (Denmark)
- FI (Finland)

## Phase 1

- Implement `build_matern_covariance_matrix`
- Implement `inject_correlated_drought_events`
- Implement `generate_price_spikes`
- Create PostgreSQL schema with partitioned tables
- Bulk-load data
