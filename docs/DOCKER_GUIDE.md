# Docker Deployment Specifications

This document outlines the strict containerization procedures necessary to execute the NIDS-ML service inside scalable deployment architectures, maintaining persistent bounds for stateful drift algorithms and active learning loops.

## 1. Image Compilation

The standard distribution utilizes Python 3.11 slim variant schemas, minimizing surface attack vectors by reducing pre-compiled executable dependencies within the distribution layer.

```bash
docker build -t nids-ml:v2.0.0 -f deployment/Dockerfile .
```

## 2. Stateless Testing Execution

Prior to production binding, developers test fundamental constraints by launching isolated containers explicitly disabling persistent layer boundaries:

```bash
docker run --rm \
  --name nids_test \
  -p 8000:8000 \
  -e MODEL_VERSION=v2.0.0 \
  -e LOG_LEVEL=DEBUG \
  nids-ml:v2.0.0
```

Validating base dependencies requires initiating GET parameter logic mapping onto the explicit `/health` structure.

## 3. Production Stateful Bindings

NIDS-ML Tier 4 components (ADWIN Drift Detection & Active Learning Feedback Buffers) require writable storage arrays mapped directly against the host file system or network-attached disk specifications. Terminating a container without volume allocations structurally obliterates drift detection variance memory and unclassified feedback samples.

### Standard Command Instantiation

```bash
docker run -d \
  --name nids_production \
  -p 8000:8000 \
  --restart unless-stopped \
  -v /var/nids/models/v2.0.0:/app/models/production:ro \
  -v /var/nids/state:/app/state:rw \
  -v /var/nids/logs:/app/logs:rw \
  -e MODEL_VERSION=v2.0.0 \
  -e STATE_DIR=/app/state \
  nids-ml:v2.0.0
```

### Explanation of Binding Directives:

- `/app/models/production:ro` executes read-only mounting, protecting underlying configuration architectures from internal structural execution failures.
- `/app/state:rw` instantiates persistent write mapping ensuring `feedback_buffer.json` logic limits scale indefinitely disregarding isolated pod instance termination behavior.
- `/app/logs:rw` allows explicit SIEM agents (e.g. FluentBit) running as parallel daemon-sets to extract bounded `.log` outputs tracking inference variables.

## 4. Deployment Compose Schemas

Orchestrating robust limits via Docker Compose requires strictly allocating system dimension availability, establishing specific scaling parameters to prevent inference queuing limitations.

```yaml
version: "3.8"

services:
  nids-inference:
    image: nids-ml:v2.0.0
    container_name: nids-inference-primary
    ports:
      - "8000:8000"
    volumes:
      - /opt/nids/models:/app/models/production:ro
      - /opt/nids/state:/app/state:rw
      - /opt/nids/logs:/app/logs:rw
    environment:
      - MODEL_VERSION=v2.0.0
      - PYTHONUNBUFFERED=1
      - WORKER_COUNT=4
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 8G
        reservations:
          cpus: "2.0"
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"]
      interval: 15s
      timeout: 3s
      retries: 4
      start_period: 20s
```

## 5. Security Contexts and Hardening

- **Non-Root Execution**: Distributions should inherently limit user execution scope logic utilizing User 1000 constraints specifically appended within the final application runtime arrays natively located inside the source `Dockerfile`.
- **Seccomp Constraints**: Custom security profiles limit syscall abstractions mapped inside Kubernetes definitions ensuring host protection structures prevent memory leak exposure patterns.
