FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy api as /app/api so "from api.data" resolves (api is the package)
COPY api /app/api

RUN useradd -u 1000 -m api
USER 1000

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
