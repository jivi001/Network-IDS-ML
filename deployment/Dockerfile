# Production Dockerfile for NIDS Inference

# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Production Image
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy Project Code
COPY setup.py .
COPY README.md .
COPY requirements.txt .
COPY nids/ ./nids/
COPY deployment/ ./deployment/
COPY configs/ ./configs/
COPY models/production/ ./models/production/

RUN pip install -e .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV MODEL_VERSION=v1.0.0

# SOC Healthcheck using CURL
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run via module path
CMD ["uvicorn", "deployment.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
