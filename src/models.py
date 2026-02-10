"""
Model wrappers for Random Forest (Tier 1) and Isolation Forest (Tier 2).
Optimized hyperparameters per the research report:
  - RF: n_estimators=200, criterion='gini', class_weight='balanced'
  - iForest: contamination='auto', n_estimators=200
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib
from typing import Optional


class SupervisedModel:
    """
    Tier 1: Random Forest classifier for known attack detection.
    Uses Gini impurity and balanced class weights.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion='gini',
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced'
        )
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train on preprocessed, SMOTE-balanced data."""
        self.model.fit(X, y)
        self.is_trained = True
        print(f"[Tier1-RF] Trained on {X.shape[0]} samples, {X.shape[1]} features")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_

    def get_classes(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.classes_

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)
        print(f"[Tier1-RF] Saved to {filepath}")

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"[Tier1-RF] Loaded from {filepath}")


class UnsupervisedModel:
    """
    Tier 2: Isolation Forest for zero-day anomaly detection.
    Trained ONLY on normal traffic to learn the baseline.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=False
        )
        self.is_trained = False
        self.threshold = None

    def train(self, X: np.ndarray):
        """Train on NORMAL traffic only."""
        self.model.fit(X)
        self.is_trained = True

        scores = self.model.decision_function(X)
        self.threshold = np.percentile(scores, 5)

        print(f"[Tier2-iForest] Trained on {X.shape[0]} normal samples")
        print(f"[Tier2-iForest] Anomaly threshold: {self.threshold:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1=normal, -1=anomaly."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (lower = more anomalous)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring")
        return self.model.decision_function(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def save(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold
        }, filepath)
        print(f"[Tier2-iForest] Saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.threshold = data['threshold']
        self.is_trained = True
        print(f"[Tier2-iForest] Loaded from {filepath}")
