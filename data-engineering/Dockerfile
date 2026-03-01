FROM apache/spark:4.0.2-python3

# Switch to root to install system deps and Python packages
USER root

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libhdf5-dev \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps needed by the Spark job executors.
# These are available on all driver + executor pods (same image).
COPY data-engineering/requirements-spark.txt /opt/spark/requirements-spark.txt
RUN pip install --no-cache-dir -r /opt/spark/requirements-spark.txt

# Copy all ingest jobs and supporting scripts into the standard Spark job path
COPY data-engineering/processing/spark_era5_ingest.py /opt/spark/job/spark_era5_ingest.py
COPY data-engineering/processing/entsoe_ingest.py     /opt/spark/job/entsoe_ingest.py
COPY data-engineering/scripts/ /opt/spark/scripts/

# Spark Operator runs the job as the spark user
USER spark
