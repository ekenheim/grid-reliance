FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generator /app

RUN useradd -m generator
USER generator

ENTRYPOINT ["python", "generate.py"]
