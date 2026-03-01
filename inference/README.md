# Inference

KubeRay RayJob for distributed HSGP inference using Nutpie. Reads from MinIO Silver (model-ready Parquet), runs NUTS sampling across Ray workers, and writes posterior traces and P10/P50/P90 artifacts to MinIO Gold.

## Planned Layout

```
inference/
├── bayesian-grid-model.yaml
└── README.md
```

## Inputs / Outputs

- **Inputs:** MinIO Silver (Parquet, spatially joined weather + grid data)
- **Outputs:** MinIO Gold (posterior traces, P10/P50/P90 wind capacity factor distributions as NetCDF)

## Run

See "How to Run" in the root [README](../README.md).
