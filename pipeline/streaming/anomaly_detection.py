"""
Bytewax anomaly detection: 6-hour sliding window computing cross-region correlation.

Phase 2: Detects when multiple regions simultaneously show low wind (correlated drought).
Emits alerts when correlation structure indicates elevated tail risk.

Usage:
    python -m bytewax.run pipeline.streaming.anomaly_detection:flow
"""

from datetime import timedelta
from typing import Any

import numpy as np

# Bytewax imports (Phase 2)
# from bytewax.dataflow import Dataflow
# from bytewax.connectors.kafka import KafkaSource, KafkaSink
# from bytewax.operators.windowing import SlidingWindower


def compute_cross_region_correlation(snapshots: list[dict]) -> dict | None:
    """
    Compute cross-region correlation from a 6-hour window of snapshots.

    Args:
        snapshots: List of spatial snapshots (each has timestamp, regions dict).

    Returns:
        Dict with timestamp, correlation_matrix, mean_wind_per_region, or None if insufficient data.
    """
    if len(snapshots) < 2:
        return None

    zone_ids = sorted(snapshots[0].get("regions", {}).keys())
    if not zone_ids:
        return None

    # Build matrix: rows = timesteps, cols = zones
    data = []
    for s in snapshots:
        row = [s.get("regions", {}).get(z, np.nan) for z in zone_ids]
        data.append(row)

    arr = np.array(data)
    if np.any(np.isnan(arr)):
        return None

    corr = np.corrcoef(arr.T)
    mean_wind = np.mean(arr, axis=0)

    return {
        "timestamp": snapshots[-1].get("timestamp"),
        "zone_ids": zone_ids,
        "correlation_matrix": corr.tolist(),
        "mean_wind_per_region": dict(zip(zone_ids, mean_wind.tolist())),
    }


# Flow definition (Phase 2 - uncomment when Bytewax/Kafka are configured)
# flow = Dataflow("anomaly_detection")
# flow.input("snapshots", KafkaSource(["redpanda:9092"], ["grid-snapshots"]))
# flow.window(SlidingWindower(timedelta(hours=6), timedelta(hours=1)))
# flow.map(compute_cross_region_correlation)
# flow.filter(lambda x: x is not None)
# flow.output("correlations", KafkaSink(["redpanda:9092"], "cross-region-correlations"))
