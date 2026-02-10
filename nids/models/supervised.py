"""
Supervised model wrapper for Random Forest (Tier 1).
Optimized hyperparameters per the research report:
  - RF: n_estimators=200, criterion='gini', class_weight='balanced'
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
