# Production Deployment Specifications

This document outlines the architectural patterns and strict operational bounds required to deploy the NIDS-ML framework in highly available SOC configurations.

## 1. Pre-Deployment Validation Constraints

Before executing image construction or infrastructure configurations, the model artifacts must resolve explicitly against rigorous continuous integration constraints:

### Artifact Generation Array

The output bound directory `models/production/<version>/` must inherently contain:

- `tier1_stacking.pkl`: Instantiated `LogisticRegression` bounds combining internal `BalancedRandomForest`, `LGBM`, and calibrated `SVC` outputs.
- `tier2_fusion.pkl`: PyTorch Tensor traces bounding the `VAEAnomalyDetector` structure.
- `preprocessor.pkl`: Configuration arrays structuring the `RobustScaler` dimensions.
- `feature_selector.pkl`: Borda count fusion masks calculating categorical bounds.
- `drift_state.json`: Internal initialization variables handling ADWIN drift variance structures.
- `feedback_buffer.json`: Initial state logic establishing active learning persistent limits.

### Minimum Evaluation Thresholds

Pipelines verify against strict precision configurations before advancing:

- True Positive calculations mandate validation mappings achieving parameters bounded consistently above F2-scores of 0.95.
- Alert Fatigue Index mapping limits strictly < 0.05 index generation rates against normal testing variations.

## 2. SIEM & Log Aggregation Protocols

NIDS-ML operationalizes directly with standard enterprise logging configurations.

### ELK / Splunk Structured Output

All standard FastAPI logic pipes directly into structural JSON representations. Concept Drift anomalies, instantiated natively via the ADWIN logic paths, immediately generate specific schema payloads targeting Elasticsearch endpoints:

```python
def forward_drift_event(drift_metadata, siem_endpoint):
    event_schema = {
        'timestamp': drift_metadata['timestamp'],
        'n_samples_at_drift': drift_metadata['n_samples_at_drift'],
        'detector_mode': drift_metadata['detector'],
        'trigger_level': 'CRITICAL'
    }
    requests.post(
        f"https://{siem_endpoint}/drift_ingest",
        json={'event': event_schema}
    )
```

## 3. Storage and State Persistence

The integration of Active Learning and Concept Drift mechanisms fundamentally rewrites the stateless scaling operations historically utilized.

### Volume Mapping

Continuous scaling distributions spanning multiple replicas mandate central external state volumes mapping persistent data elements inherently. The `FeedbackBuffer` strictly demands continuous writable access mapping to absolute system paths. Read/Write constraints explicitly scale against localized NAS (Network Attached Storage) volumes or bound EFS implementations inside AWS deployments.

## 4. Hardware Sizing & Acceleration

- **Tier 1 (Stacking Ensemble)**: High utilization spanning standard CPU vector architectures. Optimization specifies mapping multi-threaded `n_jobs=-1` constraints via Python implementations requiring roughly 1 CPU core per allocated worker.
- **Tier 2 (VAE Anomaly)**: Heavily optimized for vector processing architectures. When deploying against high-throughput network configurations exceeding 10,000 requests per minute, attaching CUDA-operable GPU components via NVIDIA hardware boundaries is explicitly recommended to process multi-tensor dimensional abstractions without buffering bottlenecks.

## 5. Health Probes and Metric Scraping

The overarching deployment must configure Liveness and Readiness probes mapping directly against internal service endpoints establishing container operability:

- **Liveness Probe**: Maps strictly to HTTP GET `/health`. Re-instantiates the container if memory parameters exceed explicit maximum logic faults.
- **Prometheus Scraping**: Exposed endpoints aggregate internal classification parameters formatting drift variance magnitudes enabling explicit metric graphs rendered via Grafana or subsequent integration layers.

## 6. Access and Authentication Management

The deployment assumes boundary protection behind internal networking firewalls. External exposition mandates configuring internal application reverse-proxies mapping structural Nginx TLS 1.3 distributions and native JWT validation methodologies against inference bounds:

```nginx
server {
    listen 443 ssl;
    server_name nids.internal.domain;

    ssl_certificate /etc/ssl/certs/nids_production.crt;
    ssl_certificate_key /etc/ssl/private/nids_production.key;
    ssl_protocols TLSv1.3;

    location /predict/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-Real-IP $remote_addr;
        limit_req zone=nids_limit burst=200 nodelay;
    }
}
```

Implementation strictly dictates adhering closely to minimum privilege specifications for external traffic generation.
