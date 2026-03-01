"""
Dagster schedules for the Grid Resilience HSGP pipeline.
"""

from dagster import ScheduleDefinition, DefaultScheduleStatus

from pipeline.dagster.jobs import grid_resilience_full_pipeline_job, grid_resilience_retrain_job


# 6-hourly retrain: every 6 hours
retrain_schedule = ScheduleDefinition(
    job=grid_resilience_retrain_job,
    cron_schedule="0 */6 * * *",  # Every 6 hours at :00
    name="grid_resilience_retrain",
    default_status=DefaultScheduleStatus.RUNNING,
)

# Full pipeline (ingest + train + forecast + alert) - daily
full_pipeline_schedule = ScheduleDefinition(
    job=grid_resilience_full_pipeline_job,
    cron_schedule="0 6 * * *",  # Daily at 06:00 UTC
    name="grid_resilience_full_pipeline",
    default_status=DefaultScheduleStatus.STOPPED,
)
