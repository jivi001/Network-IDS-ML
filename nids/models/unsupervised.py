"""
Unsupervised model wrapper for Isolation Forest (Tier 2).
Calibrated automatically against normal training data.
"""

import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


class UnsupervisedModel:
    """
    Tier 2: Isolation Forest for zero-day anomaly detection.
    Trained ONLY on normal traffic to learn the baseline.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        params = {
            'contamination': 'auto',  # 'auto' to prevent forced false positives
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'bootstrap': False
        }
        params.update(kwargs)

        self.model = IsolationForest(**params)
        self.is_fitted = False
        
        # Support ENV override for dynamic thresholding in SOC
        env_thresh = os.getenv("NIDS_ANOMALY_THRESHOLD_PCT")
        self.anomaly_percentile = float(env_thresh) if env_thresh else 1.0 
        self.threshold_ = 0.0

    def train(self, X: np.ndarray):
        """Train on NORMAL traffic only."""
        self.model.fit(X)
        self.is_fitted = True

        scores = self.model.decision_function(X)
        # Dynamic threshold based on the configured percentile
        self.threshold_ = np.percentile(scores, self.anomaly_percentile)

        print(f"[Tier2-iForest] Trained on {X.shape[0]} normal samples")
        print(f"[Tier2-iForest] Anomaly threshold: {self.threshold_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1=normal, -1=anomaly."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        scores = self.model.decision_function(X)
        return np.where(scores < self.threshold_, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before scoring")
        return self.model.decision_function(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold_
        }, filepath)
        print(f"[Tier2-iForest] Saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.threshold_ = data['threshold']
        self.is_fitted = True
        print(f"[Tier2-iForest] Loaded from {filepath}")
