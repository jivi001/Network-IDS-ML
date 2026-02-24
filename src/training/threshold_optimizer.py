from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def optimize_f2_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """Return threshold and corresponding best F2 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    beta_sq = 4.0
    f2_scores = (
        (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-12)
    )
    best_index = int(np.nanargmax(f2_scores[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
    return best_threshold, float(f2_scores[best_index])
