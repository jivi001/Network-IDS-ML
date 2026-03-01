"""
Metrics exporter for NIDS.
Implements Prometheus counters and gauges for API observability.
"""

from prometheus_client import Counter, Histogram, Gauge

# Prometheus Metrics Definition
INFERENCE_REQUEST_COUNT = Counter(
    'nids_inference_requests_total', 
    'Total number of inference requests', 
    ['tier', 'prediction', 'status']
)

INFERENCE_LATENCY = Histogram(
    'nids_inference_latency_seconds', 
    'Latency of inference requests', 
    ['tier']
)

ANOMALY_SCORE_GAUGE = Gauge(
    'nids_iforest_anomaly_score', 
    'Isolation Forest anomaly score'
)

DRIFT_SEVERITY_GAUGE = Gauge(
    'nids_drift_severity', 
    'Severity score of detected feature drift'
)

MODEL_VERSION_INFO = Gauge(
    'nids_model_version_info', 
    'Current active model version',
    ['version']
)
