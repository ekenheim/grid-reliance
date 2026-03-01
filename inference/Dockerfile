FROM python:3.11-slim-bookworm

# Security updates + build deps for JAX/NumPy/nutpie compilation
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is repo root; Dockerfile lives in inference/
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pipeline model code (needed by the remote tasks via runtime_env pip install,
# but also present in the submitter image for local imports)
COPY pipeline/__init__.py      /app/pipeline/
COPY pipeline/model/           /app/pipeline/model/
COPY pipeline/dagster/         /app/pipeline/dagster/
COPY pipeline/streaming/       /app/pipeline/streaming/

# Inference entrypoint
COPY inference/__init__.py     /app/inference/
COPY inference/run_inference.py /app/inference/run_inference.py

RUN useradd -m inference
USER inference

CMD ["python", "-m", "inference.run_inference"]
