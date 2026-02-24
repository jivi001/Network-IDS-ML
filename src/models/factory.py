from __future__ import annotations

from typing import Any

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC


def build_model(model_name: str, hyperparameters: dict[str, Any], random_state: int):
    """Build supported models for enterprise NIDS training."""
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **hyperparameters)
    if model_name == "svm":
        return SVC(probability=True, random_state=random_state, **hyperparameters)
    if model_name == "isolation_forest":
        return IsolationForest(random_state=random_state, **hyperparameters)
    raise ValueError(f"Unsupported model: {model_name}")
