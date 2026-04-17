FROM python:3.11-slim-bookworm

# Apply security updates and install build deps for JAX/NumPy.
# libopenblas-dev gives PyTensor a system BLAS to link against at JIT compile
# time; without it PyTensor warns "could not link to a BLAS installation" and
# falls back to the single-threaded NumPy C-API path, which makes HSGP SVI
# 5–10x slower than the threaded estimate.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Force PyTensor to use OpenBLAS rather than its (sometimes flaky)
# auto-detection of numpy's vendored libraries.
ENV PYTENSOR_FLAGS="blas__ldflags=-lopenblas"

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

RUN groupadd -g 1000 pipeline && useradd -u 1000 -g 1000 -m pipeline
USER 1000

CMD ["dagster", "api", "grpc", "-m", "pipeline.dagster.repository", "-h", "0.0.0.0", "-p", "4000"]
