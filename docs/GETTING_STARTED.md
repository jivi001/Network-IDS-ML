# Setup and Installation Guide

This document outlines the strict installation procedures and baseline operational execution commands for the NIDS-ML platform.

## 1. Environment Configuration

### Cloning and Virtualization

The platform must operate within an isolated virtual environment to enforce dependency strictness via Python 3.11+.

```bash
git clone https://github.com/jivi001/Network-IDS-ML.git
cd Network-IDS-ML

# Windows Initialization
python -m venv nids_env
nids_env\Scripts\activate

# Unix Initialization
python -m venv nids_env
source nids_env/bin/activate
```

### Dependency Resolution

```bash
pip install -r requirements.txt
pip install -e .
```

### Validation

Confirm the integration of standard runtime packages via module availability testing:

```bash
python -c "from nids.models.hybrid import HybridNIDS; print('Module NIDS loaded successfully.')"
```

## 2. Dataset Ingestion

The framework targets normalized `.csv` matrices mapped to standard network traffic captures. Supported ingestion architectures include NSL-KDD and UNSW-NB15.

Execute the retrieval operation targeting raw paths:

```bash
mkdir -p data/raw

# Target parameters: nsl-kdd | unsw-nb15
python scripts/download_data.py --dataset nsl-kdd
```

## 3. Training Execution

Model orchestration functions depend strictly upon explicit YAML configurations stored under `configs/training/`.

```bash
# Execute standard baseline training
python scripts/train.py --config configs/training/default.yaml
```

The pipeline operates sequentially through:

1. Feature matrix ingestion
2. Scaling and encoding (RobustScaler)
3. SHAP/Mutual Information feature subset reduction
4. Dual-tier Model training (Stacking Ensemble Tier 1 + VAE Anomaly Tier 2)
5. Metric emission calculation (MCC, F2, Precision)

Data artifacts including pickled `.pkl` serializers, feature importance thresholds, and SHAP graphical arrays are written atomically to `experiments/runs/<timestamp>/`.

## 4. Inference Validation

Inference execution pipelines instantiate the `InferencePipeline` runtime object to resolve raw input vectors against the serialized models.

```python
import sys
import numpy as np
from nids.pipelines import InferencePipeline

# Instantiate targeted historical execution model
pipeline = InferencePipeline(model_version='v1.0.0')

# Simulate 20-dimension feature vector mapping to trained shapes
sample_traffic = np.random.randn(20)

result = pipeline.predict_single(sample_traffic)

print(f"Prediction Output: {result['prediction']}")
print(f"Algorithm Confidence: {result['confidence']:.2f}")
print(f"Cascading Tier Origin: {result['tier_used']}")
print(f"Zero-Day Anomaly Scalar: {result['anomaly_score']:.4f}")
```

## 5. Dockerized Microservice Deployment

The platform is designed to scale dynamically within containerized orchestration models. The FastAPI backend exposes endpoints structured for Prometheus scraping and continuous SIEM ingestion.

```bash
# Execute container build logic
docker build -t nids:v1.0.0 -f deployment/Dockerfile .

# Orchestrate execution via compose specifications
docker-compose -f deployment/docker-compose.yml up -d
```

### Testing Endpoint Logic

```bash
# Verify internal application health flags
curl http://localhost:8000/health

# Trigger standard classification arrays
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.7, 0.9, 1.1, 0.4, 0.6, 1.3, 0.2, 1.8, 0.5, 1.0, 0.8, 1.4, 0.6, 0.9]}'
```

Output specification matches strict JSON types conforming to the expected detection schemas.
