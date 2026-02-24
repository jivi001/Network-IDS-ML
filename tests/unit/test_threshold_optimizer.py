import numpy as np

from src.training.threshold_optimizer import optimize_f2_threshold


def test_optimize_f2_threshold_range() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, f2 = optimize_f2_threshold(y_true, y_score)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f2 <= 1.0
