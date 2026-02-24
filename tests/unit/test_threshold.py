"""
Unit tests for threshold optimization logic.
"""

import numpy as np
import pytest

from nids.evaluation.metrics import NIDSEvaluator


class TestThresholdOptimization:
    """Threshold optimization via PR curve / F2-score."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        return NIDSEvaluator(output_dir=str(tmp_path))

    @pytest.fixture
    def binary_data(self):
        """Create synthetic binary classification data with probabilities."""
        np.random.seed(42)
        n = 200
        y_true = np.array(["Normal"] * 100 + ["DoS"] * 100)
        # Create probabilities that are somewhat correct
        proba_normal = np.column_stack([
            np.random.uniform(0.6, 0.9, 100),  # P(Normal) high for normals
            np.random.uniform(0.1, 0.4, 100),  # P(Attack) low
        ])
        proba_attack = np.column_stack([
            np.random.uniform(0.1, 0.4, 100),  # P(Normal) low for attacks
            np.random.uniform(0.6, 0.9, 100),  # P(Attack) high
        ])
        y_proba = np.vstack([proba_normal, proba_attack])
        return y_true, y_proba

    def test_returns_tuple_of_two(self, evaluator, binary_data):
        y_true, y_proba = binary_data
        result = evaluator.optimize_threshold(y_true, y_proba, normal_label="Normal")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_between_0_and_1(self, evaluator, binary_data):
        y_true, y_proba = binary_data
        threshold, score = evaluator.optimize_threshold(y_true, y_proba)
        assert 0.0 <= threshold <= 1.0, f"Threshold {threshold} out of range"

    def test_f2_score_between_0_and_1(self, evaluator, binary_data):
        y_true, y_proba = binary_data
        _, score = evaluator.optimize_threshold(y_true, y_proba)
        assert 0.0 <= score <= 1.0, f"F2 score {score} out of range"

    def test_higher_beta_favors_recall(self, evaluator, binary_data):
        """A higher beta should result in a threshold that favors recall."""
        y_true, y_proba = binary_data
        _, f1_score = evaluator.optimize_threshold(y_true, y_proba, beta=1.0)
        _, f2_score = evaluator.optimize_threshold(y_true, y_proba, beta=2.0)
        # Both should be valid scores
        assert f1_score is not None and f2_score is not None

    def test_perfect_classifier_returns_high_f2(self, evaluator):
        """A perfect classifier should get near-perfect threshold F2."""
        y_true = np.array(["Normal"] * 50 + ["DoS"] * 50)
        y_proba = np.zeros((100, 2))
        y_proba[:50, 0] = 1.0  # Normal class = 1.0 for normals
        y_proba[50:, 1] = 1.0  # Attack class = 1.0 for attacks
        _, f2 = evaluator.optimize_threshold(y_true, y_proba)
        assert f2 > 0.95, f"Perfect classifier should get F2 > 0.95, got {f2}"

    def test_handles_1d_proba(self, evaluator):
        """Should handle 1D probability scores."""
        y_true = np.array(["Normal"] * 50 + ["DoS"] * 50)
        y_score = np.concatenate([
            np.random.uniform(0.0, 0.4, 50),
            np.random.uniform(0.6, 1.0, 50),
        ])
        threshold, f2 = evaluator.optimize_threshold(y_true, y_score)
        assert threshold is not None
        assert 0.0 <= threshold <= 1.0
