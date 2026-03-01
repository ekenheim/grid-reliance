# Infrastructure Alignment (Homelab / home-ops-upgrade)

This document aligns the Pan-European Grid Resilience app repo with the **home-ops-upgrade** (Talos Linux / Flux GitOps) homelab. It summarizes where storage, compute, and credentials live in that cluster so this repo’s scripts and docs stay consistent.

## Storage: Ceph RGW Bronze / Silver / Gold

The cluster uses **Ceph RGW** (not MinIO) for the data lakehouse. Three ObjectBucketClaims provision:

| Tier   | Bucket name               | Purpose |
|--------|---------------------------|--------|
| Bronze | `grid-resilience-bronze`  | Raw ERA5 NetCDF, ENTSO-E XML |
| Silver | `grid-resilience-silver`  | Parquet, spatially joined, model-ready |
| Gold   | `grid-resilience-gold`    | P10/P50/P90 posterior traces (NetCDF) |

Rook creates a **Secret** (e.g. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) and **ConfigMap** (BUCKET_HOST, BUCKET_NAME, BUCKET_PORT, BUCKET_REGION) in the `datasci` namespace for each claim. Use these for Spark, Ray, and any job that reads/writes Bronze, Silver, or Gold.

Local development can keep using MinIO (`docker compose`) with the same bucket names or prefixes.

## Compute

- **Ray:** KubeRay cluster in `datasci`; head service `ray-kuberay-head-svc.datasci.svc.cluster.local:10001`. Resources were increased (head 4 CPU / 8Gi, workers up to 8 CPU / 16Gi, max 12 workers).
- **Dagster:** `RAY_ADDRESS` is set so run launcher pods use the existing Ray cluster (no per-run Ray cluster).
- **Spark:** Standalone (Bitnami) plus **kubeflow Spark Operator** for SparkApplication CRDs. Worker CPU was increased (4–8 CPU). Spark PVCs (e.g. `spark-work-dir`, `spark-checkpoint-pvc`) were resized (e.g. 500Gi / 100Gi) for ERA5 volumes.

## Credentials (to be done)

- **ECMWF CDS API key:** Store in **ExternalSecrets** and inject into the ERA5 fetch job (or Spark driver) so the cluster can pull ERA5 without a local `~/.cdsapirc`.
- **ENTSO-E security token:** Same pattern when ENTSO-E ingest is added.

## Spark Operator (kubeflow)

The app’s ingest pipelines assume **SparkApplication** CRDs (`apiVersion: sparkoperator.k8s.io/v1beta2`). These are provided by the **kubeflow Spark Operator**, not the Bitnami Spark chart. The OCI Helm URL for the kubeflow operator returned 403 at reconciliation; switching the HelmRepository to the HTTP repo (`https://kubeflow.github.io/spark-operator`) is still to be done.

## Still to do (from infra side)

- Fix kubeflow Spark Operator Helm source (HTTP repo instead of OCI if needed).
- Add ExternalSecrets for ECMWF API key and ENTSO-E token.
- Optionally: custom Ray image (Harbor) with PyMC, Nutpie, ArviZ pre-installed.
- Deploy Streamlit risk dashboard.
- Restore worker nodes to reach full 94-core capacity.

## This repo’s usage

- **ERA5 fetch:** `data-engineering/scripts/fetch_era5_nordic.py` can target Bronze via env vars; in-cluster, set `BRONZE_ENDPOINT`, `BRONZE_ACCESS_KEY`, `BRONZE_SECRET_KEY`, `BRONZE_BUCKET` from the Ceph RGW Secret/ConfigMap (or from ExternalSecrets).
- **Pipeline / API:** In production, point MinIO-compatible client config at Ceph RGW and the Silver/Gold bucket names above; locally keep using MinIO and `.env` as today.
