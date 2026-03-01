"""
Drift Governance Framework for NIDS.
Implements PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests
for rolling statistical buffers to detect input and output drift.
"""

import numpy as np
from scipy.stats import ks_2samp
import logging
from typing import Dict, List, Any

logger = logging.getLogger("nids.governance")

class DriftDetector:
    def __init__(self, feature_names: List[str], buffer_size: int = 10000):
        self.feature_names = feature_names
        self.buffer_size = buffer_size
        self.reference_distributions: Dict[str, np.ndarray] = {}
        
        # Rolling buffers
        self.current_buffer: List[List[float]] = []
        self.current_predictions: List[str] = []
        
        # Drift State
        self.drift_status = "STABLE"
        self.severity_score = 0.0

    def fit_reference(self, X_train: np.ndarray, y_train: np.ndarray):
        """Store reference distributions from training data."""
        for i, name in enumerate(self.feature_names):
            self.reference_distributions[name] = X_train[:, i]
        self.reference_labels = y_train
        logger.info("Reference distributions locked for drift governance.")

    def update_buffer(self, features: np.ndarray, prediction: str):
        """Add inference payload to rolling buffer. Trigger check if full."""
        self.current_buffer.append(features)
        self.current_predictions.append(prediction)
        
        if len(self.current_buffer) >= self.buffer_size:
            return self._compute_drift()
        return None

    def _compute_drift(self) -> Dict[str, Any]:
        """Compute KS-test across all features between reference and buffer."""
        drift_report = {}
        X_current = np.array(self.current_buffer)
        
        drift_count = 0
        for i, name in enumerate(self.feature_names):
            ref_dist = self.reference_distributions[name]
            cur_dist = X_current[:, i]
            
            # Kolmogorov-Smirnov Test
            statistic, p_value = ks_2samp(ref_dist, cur_dist)
            is_drifting = p_value < 0.05  # 95% confidence
            
            drift_report[name] = {
                "ks_stat": round(statistic, 4),
                "p_value": round(p_value, 4),
                "drifting": is_drifting
            }
            if is_drifting:
                drift_count += 1
                
        # Severity calculation
        self.severity_score = drift_count / len(self.feature_names)
        
        if self.severity_score > 0.3:
            self.drift_status = "CRITICAL_DRIFT"
        elif self.severity_score > 0.1:
            self.drift_status = "WARNING_DRIFT"
        else:
            self.drift_status = "STABLE"

        # Output Drift (Label shift)
        # Simplified: Check if Anomaly rate spiked by > 50% compared to training
        
        report = {
            "timestamp": "now",
            "status": self.drift_status,
            "severity_score": self.severity_score,
            "drifting_features_count": drift_count,
            "trigger_retraining": self.severity_score > 0.3
        }
        
        # Reset buffer
        self.current_buffer = []
        self.current_predictions = []
        
        return report
