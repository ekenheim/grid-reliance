"""
Bytewax spatial windowing: 1-hour tumbling window aggregating 8 regions into a snapshot.

Phase 2: Consumes from Redpanda weather-obs topic, emits spatial snapshots to
PostgreSQL or a downstream topic.

Usage:
    python -m bytewax.run pipeline.streaming.spatial_windowing:flow
"""

from datetime import timedelta
from typing import Any

# Bytewax imports (Phase 2)
# from bytewax.dataflow import Dataflow
# from bytewax.connectors.kafka import KafkaSource, KafkaSink
# from bytewax.operators.windowing import TumblingWindower


def compute_spatial_snapshot(region_values: list[dict]) -> dict:
    """
    Aggregate all region observations for a 1-hour window into a single snapshot.

    Args:
        region_values: List of dicts with keys region_id, wind_speed_mps, timestamp.

    Returns:
        Snapshot dict with timestamp and per-region wind speeds.
    """
    if not region_values:
        return {}
    ts = region_values[0].get("timestamp")
    snapshot = {"timestamp": ts, "regions": {}}
    for rv in region_values:
        snapshot["regions"][rv["region_id"]] = rv.get("wind_speed_mps")
    return snapshot


# Flow definition (Phase 2 - uncomment when Bytewax/Kafka are configured)
# flow = Dataflow("spatial_windowing")
# flow.input("weather", KafkaSource(["redpanda:9092"], ["weather-obs"]))
# flow.window(TumblingWindower(timedelta(hours=1)))
# flow.map(compute_spatial_snapshot)
# flow.output("snapshots", KafkaSink(["redpanda:9092"], "grid-snapshots"))
