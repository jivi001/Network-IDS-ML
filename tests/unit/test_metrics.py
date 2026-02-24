import numpy as np

from src.evaluation.metrics import compute_binary_metrics


def test_compute_binary_metrics_contains_security_fields() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.7, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    assert "false_positive_rate" in metrics
    assert "per_class_recall" in metrics
    assert len(metrics["confusion_matrix"]) == 2
