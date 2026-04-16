# Custom Ray worker image for grid-resilience-inference.
#
# Extends the exact Ray base image running on the ray-kuberay cluster
# (rayproject/ray:2.54.0, Python 3.10, conda base) with the packages
# required for HSGP chain sampling.
#
# The platform team points the Ray HelmRelease at this image:
#   workerGroupSpecs[0].template.spec.containers[0].image:
#     ghcr.io/ekenheim/grid-resilience-ray-worker:latest
#
# This avoids the fragile runtime_env pip install against the conda base
# (zarr/nutpie resolution failures with pandas 1.5.3 and no zarr pre-installed).
FROM rayproject/ray:2.54.0

USER root

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY ray-worker/requirements-worker.txt /tmp/requirements-worker.txt

# nutpie >=0.16 requires zarr >=3.1.0 which requires Python >=3.11.
# This base image ships Python 3.10, so nutpie cannot be installed here.
# numpyro (JAX-backed) is used instead — equally fast, no zarr dependency.
RUN pip install --no-cache-dir -r /tmp/requirements-worker.txt

# Copy pipeline model code so workers can import it directly
# (avoids a second pip install of the pipeline package at runtime).
COPY pipeline/__init__.py      /home/ray/pipeline/
COPY pipeline/model/           /home/ray/pipeline/model/

# Set PYTHONPATH without extending an undefined variable (avoids BuildKit warning)
ENV PYTHONPATH=/home/ray

# ray user is UID 1000 in rayproject/ray base
USER 1000
