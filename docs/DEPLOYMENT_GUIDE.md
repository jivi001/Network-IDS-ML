# 🚀 Production Deployment Guide

This guide covers deploying the NIDS system to production environments.

---

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Integration with SOC](#integration-with-soc)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 1. Deployment Options

### Option A: Local Python Service
- **Best for**: Development, testing, small-scale deployment
- **Pros**: Simple setup, easy debugging
- **Cons**: Manual dependency management, no isolation

### Option B: Docker Container
- **Best for**: Production, scalable deployment
- **Pros**: Isolated environment, easy scaling, reproducible
- **Cons**: Requires Docker knowledge

### Option C: Cloud Platform
- **Best for**: Enterprise, high-availability requirements
- **Pros**: Auto-scaling, managed infrastructure, high availability
- **Cons**: Higher cost, vendor lock-in

---

## 2. Pre-Deployment Checklist

### ✅ Model Training

- [ ] Train model on representative dataset
- [ ] Achieve acceptable performance metrics (Recall > 90%)
- [ ] Test on holdout dataset
- [ ] Perform cross-dataset validation
- [ ] Save model to `models/production/v1.0.0/`

### ✅ Model Artifacts

Ensure these files exist:
```
models/production/v1.0.0/
├── tier1_rf.pkl           # Random Forest model
├── tier2_iforest.pkl      # Isolation Forest model
├── preprocessor.pkl       # Data preprocessor
└── feature_selector.pkl   # Feature selector
```

### ✅ Configuration

- [ ] Review `configs/training/default.yaml`
- [ ] Set correct `normal_label` for your dataset
- [ ] Configure contamination rate for Tier 2
- [ ] Set appropriate resource limits

### ✅ Testing

- [ ] Test inference pipeline locally
- [ ] Verify prediction accuracy
- [ ] Load test API endpoints
- [ ] Test error handling

### ✅ Security

- [ ] Add authentication to API (if needed)
- [ ] Set up HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Review access controls

---

## 3. Local Deployment

### Install Dependencies

```bash
# Create virtual environment
python -m venv nids_env
source nids_env/bin/activate  # Windows: nids_env\Scripts\activate

# Install package
pip install -e .
```

### Run Inference API

```bash
# Start the API server
python deployment/inference_api.py

# Server runs on http://localhost:8000
```

### Test Locally

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

### Run as Background Service (Linux)

Create systemd service file `/etc/systemd/system/nids.service`:

```ini
[Unit]
Description=NIDS Inference Service
After=network.target

[Service]
Type=simple
User=nids
WorkingDirectory=/opt/nids
Environment="PATH=/opt/nids/nids_env/bin"
ExecStart=/opt/nids/nids_env/bin/python deployment/inference_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nids
sudo systemctl start nids
sudo systemctl status nids
```

---

## 4. Docker Deployment

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed Docker instructions.

### Quick Start

```bash
# Build image
docker build -t nids:v1.0.0 -f deployment/Dockerfile .

# Run container
docker-compose -f deployment/docker-compose.yml up -d

# Test
curl http://localhost:8000/health
```

---

## 5. Cloud Deployment

### AWS Deployment

#### Option 1: EC2 Instance

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# 3. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# 4. Clone repository
git clone https://github.com/yourusername/Network-IDS-ML.git
cd Network-IDS-ML

# 5. Build and run
docker build -t nids:v1.0.0 -f deployment/Dockerfile .
docker run -d -p 8000:8000 --name nids nids:v1.0.0

# 6. Configure security group to allow port 8000
```

#### Option 2: ECS (Elastic Container Service)

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name nids
docker tag nids:v1.0.0 ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/nids:v1.0.0
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/nids:v1.0.0

# 2. Create ECS task definition (JSON)
# 3. Create ECS service
# 4. Deploy to ECS cluster
```

### Google Cloud Deployment

#### Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/nids:v1.0.0

# 2. Deploy to Cloud Run
gcloud run deploy nids \
  --image gcr.io/PROJECT_ID/nids:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10

# 3. Get service URL
gcloud run services describe nids --region us-central1
```

### Azure Deployment

#### Container Instances

```bash
# 1. Create resource group
az group create --name nids-rg --location eastus

# 2. Create container registry
az acr create --resource-group nids-rg --name nidsregistry --sku Basic

# 3. Push image
az acr login --name nidsregistry
docker tag nids:v1.0.0 nidsregistry.azurecr.io/nids:v1.0.0
docker push nidsregistry.azurecr.io/nids:v1.0.0

# 4. Deploy container
az container create \
  --resource-group nids-rg \
  --name nids-inference \
  --image nidsregistry.azurecr.io/nids:v1.0.0 \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

---

## 6. Integration with SOC

### REST API Integration

The NIDS exposes a REST API that can be integrated into SOC workflows:

```python
import requests

# SOC monitoring script
def analyze_network_traffic(traffic_features):
    """
    Analyze network traffic using NIDS API.
    
    Args:
        traffic_features: List of 20 numerical features
    
    Returns:
        dict: Prediction result
    """
    response = requests.post(
        'http://nids-server:8000/predict',
        json={'features': traffic_features},
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Alert if attack detected
        if result['prediction'] != 'Normal':
            send_alert(
                severity='HIGH',
                attack_type=result['prediction'],
                confidence=result['confidence'],
                tier=result['tier_used']
            )
        
        return result
    else:
        raise Exception(f"NIDS API error: {response.status_code}")

# Example usage
traffic = [0.5, 1.2, 0.8, ...]  # 20 features
prediction = analyze_network_traffic(traffic)
print(f"Prediction: {prediction['prediction']}")
```

### SIEM Integration

#### Splunk Integration

```python
# Splunk forwarder script
import requests
import json

def forward_to_splunk(nids_result, traffic_metadata):
    """Forward NIDS results to Splunk."""
    event = {
        'timestamp': traffic_metadata['timestamp'],
        'source_ip': traffic_metadata['src_ip'],
        'dest_ip': traffic_metadata['dst_ip'],
        'prediction': nids_result['prediction'],
        'confidence': nids_result['confidence'],
        'tier': nids_result['tier_used'],
        'anomaly_score': nids_result['anomaly_score']
    }
    
    # Send to Splunk HTTP Event Collector
    requests.post(
        'https://splunk-server:8088/services/collector',
        headers={'Authorization': 'Splunk YOUR_TOKEN'},
        json={'event': event}
    )
```

#### ELK Stack Integration

```python
from elasticsearch import Elasticsearch

# Index NIDS results in Elasticsearch
es = Elasticsearch(['http://elasticsearch:9200'])

def index_nids_result(nids_result, traffic_metadata):
    """Index NIDS result in Elasticsearch."""
    doc = {
        '@timestamp': traffic_metadata['timestamp'],
        'source': traffic_metadata['src_ip'],
        'destination': traffic_metadata['dst_ip'],
        'nids': {
            'prediction': nids_result['prediction'],
            'confidence': nids_result['confidence'],
            'tier': nids_result['tier_used'],
            'anomaly_score': nids_result['anomaly_score']
        }
    }
    
    es.index(index='nids-alerts', document=doc)
```

---

## 7. Monitoring & Maintenance

### Health Monitoring

Set up automated health checks:

```bash
# Cron job (every 5 minutes)
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart nids
```

### Performance Monitoring

Monitor key metrics:
- **Response time**: < 100ms for single prediction
- **Throughput**: Predictions per second
- **Error rate**: < 0.1%
- **Memory usage**: < 4GB
- **CPU usage**: < 80%

### Logging

Configure structured logging:

```python
# In deployment/inference_api.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/nids.log'),
        logging.StreamHandler()
    ]
)
```

### Model Updates

Deploy new model versions:

```bash
# 1. Train new model
python scripts/train.py --config configs/training/default.yaml

# 2. Copy to production with new version
mkdir -p models/production/v1.1.0
cp experiments/runs/NEW_EXPERIMENT/models/* models/production/v1.1.0/

# 3. Update environment variable
docker run -d \
  --name nids-inference-v1.1 \
  -p 8001:8000 \
  -e MODEL_VERSION=v1.1.0 \
  nids:v1.0.0

# 4. Test new version
curl http://localhost:8001/health

# 5. Switch traffic (update load balancer)
# 6. Stop old version
docker stop nids-inference
```

### Backup & Recovery

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/production/

# Backup configs
tar -czf configs-backup-$(date +%Y%m%d).tar.gz configs/

# Store backups in S3/GCS/Azure Blob
aws s3 cp models-backup-*.tar.gz s3://your-bucket/backups/
```

---

## 🔒 Security Best Practices

### 1. API Authentication

Add authentication to the API:

```python
# In deployment/inference_api.py
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... prediction logic
```

### 2. HTTPS/TLS

Use reverse proxy (nginx) with SSL:

```nginx
server {
    listen 443 ssl;
    server_name nids.example.com;
    
    ssl_certificate /etc/ssl/certs/nids.crt;
    ssl_certificate_key /etc/ssl/private/nids.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Rate Limiting

Prevent abuse:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per minute")
def predict():
    # ... prediction logic
```

---

## ✅ Deployment Checklist

Before going live:

- [ ] Model trained and validated
- [ ] All artifacts in `models/production/`
- [ ] Docker image built and tested
- [ ] Health checks configured
- [ ] Monitoring set up
- [ ] Logging configured
- [ ] Backups automated
- [ ] Security measures implemented
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

---

For Docker-specific deployment, see [DOCKER_GUIDE.md](DOCKER_GUIDE.md).  
For training new models, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).
