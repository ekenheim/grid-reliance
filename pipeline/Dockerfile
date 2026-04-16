FROM python:3.11-slim-bookworm

# Apply security updates and install build deps for JAX/NumPy
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is repo root; Dockerfile lives in pipeline/
COPY pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY pipeline/__init__.py /app/pipeline/
COPY pipeline/dagster/ /app/pipeline/dagster/
COPY pipeline/model/ /app/pipeline/model/
COPY pipeline/streaming/ /app/pipeline/streaming/

# Fetch scripts (ERA5 + ENTSO-E) used by the bronze fetch Dagster assets
COPY data-engineering/scripts/ /app/data-engineering/scripts/

ENV DAGSTER_HOME=/opt/dagster
RUN mkdir -p /opt/dagster && chmod 777 /opt/dagster

RUN useradd -u 1000 -m pipeline
USER 1000

CMD ["dagster", "api", "grpc", "-m", "pipeline.dagster.repository", "-h", "0.0.0.0", "-p", "4000"]
