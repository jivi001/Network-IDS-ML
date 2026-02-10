# 🐳 Docker Deployment Guide

This guide explains how to build and deploy the NIDS system using Docker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build Docker Image](#build-docker-image)
3. [Run Container](#run-container)
4. [Test Deployment](#test-deployment)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### Install Docker

**Windows:**
- Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Install and restart your computer
- Verify: `docker --version`

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Verify
docker --version
docker-compose --version
```

**macOS:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Install and start Docker Desktop
- Verify: `docker --version`

### Prepare Trained Model

Before building the Docker image, you need a trained model:

```bash
# Train a model first
python scripts/train.py --config configs/training/default.yaml

# Copy trained model to production directory
mkdir -p models/production/v1.0.0
cp experiments/runs/YOUR_EXPERIMENT_ID/models/* models/production/v1.0.0/
```

---

## 2. Build Docker Image

### Basic Build

```bash
# Build the Docker image
docker build -t nids:v1.0.0 -f deployment/Dockerfile .
```

### Build with Custom Tag

```bash
# Build with your own tag
docker build -t your-registry/nids:latest -f deployment/Dockerfile .
```

### Verify Build

```bash
# List Docker images
docker images | grep nids

# Output:
# nids    v1.0.0    abc123def456    2 minutes ago    1.2GB
```

---

## 3. Run Container

### Using Docker Run

```bash
# Run the container
docker run -d \
  --name nids-inference \
  -p 8000:8000 \
  -v $(pwd)/models/production:/app/models/production:ro \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_VERSION=v1.0.0 \
  -e LOG_LEVEL=INFO \
  nids:v1.0.0
```

**Windows PowerShell:**
```powershell
docker run -d `
  --name nids-inference `
  -p 8000:8000 `
  -v ${PWD}/models/production:/app/models/production:ro `
  -v ${PWD}/logs:/app/logs `
  -e MODEL_VERSION=v1.0.0 `
  -e LOG_LEVEL=INFO `
  nids:v1.0.0
```

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop the service
docker-compose -f deployment/docker-compose.yml down
```

### Check Container Status

```bash
# List running containers
docker ps

# View container logs
docker logs nids-inference

# Follow logs in real-time
docker logs -f nids-inference
```

---

## 4. Test Deployment

### Health Check

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","model_version":"v1.0.0"}
```

### Single Prediction

```bash
# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.7, 0.9, 1.1, 0.4,
                 0.6, 1.3, 0.2, 1.8, 0.5, 1.0, 0.8, 1.4, 0.6, 0.9]
  }'

# Expected output:
# {
#   "prediction": "Normal",
#   "confidence": 0.95,
#   "tier_used": 2,
#   "anomaly_score": 0.12
# }
```

**Windows PowerShell:**
```powershell
$body = @{
    features = @(0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.7, 0.9, 1.1, 0.4,
                 0.6, 1.3, 0.2, 1.8, 0.5, 1.0, 0.8, 1.4, 0.6, 0.9)
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.7, 0.9, 1.1, 0.4,
       0.6, 1.3, 0.2, 1.8, 0.5, 1.0, 0.8, 1.4, 0.6, 0.9],
      [1.5, 0.2, 1.8, 0.1, 2.3, 0.5, 1.7, 0.3, 2.1, 0.8,
       1.6, 0.4, 2.2, 0.6, 1.9, 0.7, 2.0, 0.5, 1.8, 0.9]
    ]
  }'
```

---

## 5. Production Deployment

### Environment Variables

Configure the container using environment variables:

```bash
docker run -d \
  --name nids-inference \
  -p 8000:8000 \
  -e MODEL_VERSION=v1.0.0 \
  -e LOG_LEVEL=INFO \
  -e PYTHONUNBUFFERED=1 \
  nids:v1.0.0
```

**Available Variables:**
- `MODEL_VERSION`: Model version to load (default: v1.0.0)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONUNBUFFERED`: Set to 1 for real-time logs

### Volume Mounts

Mount directories for persistence:

```bash
docker run -d \
  --name nids-inference \
  -p 8000:8000 \
  -v /path/to/models:/app/models/production:ro \  # Read-only models
  -v /path/to/logs:/app/logs \                     # Writable logs
  nids:v1.0.0
```

### Resource Limits

Set CPU and memory limits:

```bash
docker run -d \
  --name nids-inference \
  -p 8000:8000 \
  --cpus="2.0" \
  --memory="4g" \
  nids:v1.0.0
```

### Restart Policy

Auto-restart on failure:

```bash
docker run -d \
  --name nids-inference \
  -p 8000:8000 \
  --restart unless-stopped \
  nids:v1.0.0
```

### Docker Compose Production Config

Edit `deployment/docker-compose.yml`:

```yaml
version: '3.8'

services:
  nids-inference:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    container_name: nids-inference
    ports:
      - "8000:8000"
    volumes:
      - ../models/production:/app/models/production:ro
      - ../logs:/app/logs
    environment:
      - MODEL_VERSION=v1.0.0
      - LOG_LEVEL=INFO
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s
```

---

## 6. Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs nids-inference
```

**Common issues:**
- Model files not found → Check volume mounts
- Port already in use → Change port mapping
- Out of memory → Increase memory limit

### Health Check Failing

**Test manually:**
```bash
# Enter container
docker exec -it nids-inference bash

# Test health endpoint from inside
curl http://localhost:8000/health
```

### Prediction Errors

**Check model version:**
```bash
# Verify model files exist
docker exec nids-inference ls -la /app/models/production/v1.0.0/

# Expected files:
# tier1_rf.pkl
# tier2_iforest.pkl
# preprocessor.pkl
# feature_selector.pkl
```

### Performance Issues

**Monitor resources:**
```bash
# Check container stats
docker stats nids-inference

# Output shows CPU%, MEM%, NET I/O
```

**Solutions:**
- Increase CPU/memory limits
- Use batch prediction for multiple samples
- Optimize model (reduce n_estimators)

---

## 🚀 Advanced Deployment

### Push to Docker Registry

```bash
# Tag image
docker tag nids:v1.0.0 your-registry.com/nids:v1.0.0

# Login to registry
docker login your-registry.com

# Push image
docker push your-registry.com/nids:v1.0.0
```

### Deploy to Cloud

**AWS ECS:**
1. Push image to Amazon ECR
2. Create ECS task definition
3. Deploy to ECS cluster

**Google Cloud Run:**
```bash
# Tag for GCR
docker tag nids:v1.0.0 gcr.io/PROJECT_ID/nids:v1.0.0

# Push to GCR
docker push gcr.io/PROJECT_ID/nids:v1.0.0

# Deploy to Cloud Run
gcloud run deploy nids \
  --image gcr.io/PROJECT_ID/nids:v1.0.0 \
  --platform managed \
  --port 8000
```

**Azure Container Instances:**
```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name nids-inference \
  --image your-registry.com/nids:v1.0.0 \
  --ports 8000 \
  --cpu 2 \
  --memory 4
```

---

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
docker logs -f nids-inference

# Last 100 lines
docker logs --tail 100 nids-inference

# Logs since 1 hour ago
docker logs --since 1h nids-inference
```

### Health Monitoring

Set up automated health checks:

```bash
# Cron job to check health every 5 minutes
*/5 * * * * curl -f http://localhost:8000/health || echo "NIDS health check failed" | mail -s "Alert" admin@example.com
```

---

## ✅ Next Steps

After deployment:
1. **Monitor**: Set up logging and alerting
2. **Scale**: Deploy multiple instances behind load balancer
3. **Update**: Deploy new model versions with zero downtime
4. **Secure**: Add authentication and HTTPS

For production best practices, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
