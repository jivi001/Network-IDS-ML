"""
Unit tests for InferencePipeline.
Uses a fully trained in-memory pipeline to test prediction contracts.
"""

import numpy as np
import pandas as pd
import pytest

from nids.preprocessing.preprocessor import NIDSPreprocessor
from nids.features.selection import FeatureSelector
from nids.models.hybrid import HybridNIDS


# ─── Fixture: minimal trained pipeline components ───────────────────────────

@pytest.fixture(scope="module")
def tiny_dataset():
    """Return a small but class-balanced dataset for fast testing."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "duration": np.random.uniform(0, 10, n),
        "src_bytes": np.random.uniform(0, 1000, n),
        "dst_bytes": np.random.uniform(0, 1000, n),
        "land": np.random.randint(0, 2, n).astype(float),
        "wrong_fragment": np.random.randint(0, 3, n).astype(float),
    })
    y = np.array(["Normal"] * 40 + ["DoS"] * 20)
    return X, y


@pytest.fixture(scope="module")
def trained_components(tiny_dataset):
    X, y = tiny_dataset

    preprocessor = NIDSPreprocessor(random_state=42)
    X_proc = preprocessor.fit_transform(X)

    selector = FeatureSelector(n_features=min(3, X_proc.shape[1]), method="importance", random_state=42)
    feature_names = preprocessor.get_feature_names()
    X_sel = selector.fit_transform(X_proc, y, feature_names)

    model = HybridNIDS(random_state=42)
    model.train(X_sel, y, normal_label="Normal")

    return preprocessor, selector, model, X_sel, y


# ─── Tests: HybridNIDS prediction contract ──────────────────────────────────

class TestHybridNIDSPrediction:
    def test_predict_returns_two_arrays(self, trained_components):
        _, _, model, X_sel, _ = trained_components
        preds, tiers = model.predict(X_sel[:5])
        assert preds.shape == (5,), "Should return one label per sample"
        assert tiers.shape == (5,), "Should return one tier flag per sample"

    def test_predict_only_valid_labels(self, trained_components):
        _, _, model, X_sel, _ = trained_components
        valid = {"Normal", "DoS", "Zero_Day_Anomaly"}
        preds, _ = model.predict(X_sel)
        assert set(preds).issubset(valid), f"Unexpected labels: {set(preds) - valid}"

    def test_tier_flags_are_1_or_2(self, trained_components):
        _, _, model, X_sel, _ = trained_components
        _, tiers = model.predict(X_sel)
        assert set(tiers.tolist()).issubset({1, 2}), f"Invalid tier flags: {set(tiers)}"

    def test_predict_with_scores_keys(self, trained_components):
        _, _, model, X_sel, _ = trained_components
        result = model.predict_with_scores(X_sel[:3])
        required_keys = {
            "final_predictions", "tier_used",
            "tier1_probabilities", "tier2_anomaly_scores"
        }
        assert required_keys.issubset(result.keys())

    def test_confidence_in_0_1_range(self, trained_components):
        _, _, model, X_sel, _ = trained_components
        result = model.predict_with_scores(X_sel[:10])
        proba = result["tier1_probabilities"]
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_untrained_raises(self):
        model = HybridNIDS()
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(np.zeros((1, 3)))


# ─── Tests: edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_sample_prediction(self, trained_components):
        preprocessor, selector, model, _, _ = trained_components
        single = pd.DataFrame({
            "duration": [0.0],
            "src_bytes": [0.0],
            "dst_bytes": [0.0],
            "land": [0.0],
            "wrong_fragment": [0.0],
        })
        X_proc = preprocessor.transform(single)
        X_sel = selector.transform(X_proc)
        preds, tiers = model.predict(X_sel)
        assert len(preds) == 1

    def test_extreme_values_dont_crash(self, trained_components):
        preprocessor, selector, model, _, _ = trained_components
        extreme = pd.DataFrame({
            "duration": [1e9],
            "src_bytes": [1e9],
            "dst_bytes": [0.0],
            "land": [1.0],
            "wrong_fragment": [0.0],
        })
        X_proc = preprocessor.transform(extreme)
        X_sel = selector.transform(X_proc)
        preds, _ = model.predict(X_sel)
        assert len(preds) == 1

    def test_null_handling_in_preprocessor(self, trained_components):
        preprocessor, selector, model, _, _ = trained_components
        with_null = pd.DataFrame({
            "duration": [np.nan],
            "src_bytes": [500.0],
            "dst_bytes": [100.0],
            "land": [0.0],
            "wrong_fragment": [0.0],
        })
        # Should not raise, NaN should be handled by imputer
        X_proc = preprocessor.transform(with_null)
        assert not np.any(np.isnan(X_proc))
