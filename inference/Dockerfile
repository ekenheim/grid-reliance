FROM python:3.11-slim-bookworm

# Security updates + build deps for JAX/NumPy/nutpie compilation.
# libopenblas-dev gives PyTensor a system BLAS to link against at JIT compile
# time. The submitter imports pymc/build_hsgp_model for posterior predictive
# sampling, so the BLAS link matters here too — not just on the Ray workers.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTENSOR_FLAGS="blas__ldflags=-lopenblas"

WORKDIR /app

# Build context is repo root; Dockerfile lives in inference/
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pipeline model code — imported by the submitter pod (driver) and also baked
# into the ray-worker image so workers can import it without a runtime_env.
COPY pipeline/__init__.py      /app/pipeline/
COPY pipeline/model/           /app/pipeline/model/
COPY pipeline/dagster/         /app/pipeline/dagster/
COPY pipeline/streaming/       /app/pipeline/streaming/

# Inference entrypoint
COPY inference/__init__.py     /app/inference/
COPY inference/run_inference.py /app/inference/run_inference.py

RUN groupadd -g 1000 inference && useradd -u 1000 -g 1000 -m inference
USER 1000

CMD ["python", "-m", "inference.run_inference"]
