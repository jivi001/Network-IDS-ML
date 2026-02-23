"""
Unit tests for InferencePipeline.
These tests use mocked model files to avoid disk I/O.
"""

import numpy as np
import pandas as pd
import pytest
import joblib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from nids.preprocessing.preprocessor import NIDSPreprocessor
from nids.features.selection import FeatureSelector
from nids.models.hybrid import HybridNIDS
from nids.pipelines.inference import InferencePipeline


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_and_save_tiny_model(tmpdir: Path):
    """Train a tiny model and save all artifacts to tmpdir. Returns the dir."""
    np.random.seed(0)
    n = 60
    X = pd.DataFrame({
        "duration":       np.random.uniform(0, 10, n),
        "src_bytes":      np.random.uniform(0, 1000, n),
        "dst_bytes":      np.random.uniform(0, 1000, n),
        "land":           np.zeros(n),
        "wrong_fragment": np.zeros(n),
    })
    y = np.array(["Normal"] * 40 + ["DoS"] * 20)

    preprocessor = NIDSPreprocessor(random_state=0)
    X_proc = preprocessor.fit_transform(X)

    feat_names = preprocessor.get_feature_names()
    selector = FeatureSelector(
        n_features=min(3, X_proc.shape[1]),
        method="importance",
        random_state=0,
    )
    X_sel = selector.fit_transform(X_proc, y, feat_names)

    model = HybridNIDS(random_state=0)
    model.train(X_sel, y, normal_label="Normal")

    model.save(str(tmpdir / "tier1_rf.pkl"), str(tmpdir / "tier2_iforest.pkl"))
    joblib.dump(preprocessor, tmpdir / "preprocessor.pkl")
    joblib.dump(selector, tmpdir / "feature_selector.pkl")

    return X, preprocessor, selector


# ─── Fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_and_sample():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        X_raw, preprocessor, selector = _build_and_save_tiny_model(tmpdir)
        pipeline = InferencePipeline(model_dir=str(tmpdir))
        sample_row = X_raw.iloc[[0]]
        yield pipeline, sample_row, preprocessor


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestInferencePipeline:
    def test_loads_without_error(self, pipeline_and_sample):
        pipeline, _, _ = pipeline_and_sample
        assert pipeline.model is not None
        assert pipeline.preprocessor is not None
        assert pipeline.selector is not None

    def test_predict_single_returns_dict(self, pipeline_and_sample):
        pipeline, sample, _ = pipeline_and_sample
        result = pipeline.predict_single(sample)
        assert isinstance(result, dict)

    def test_predict_single_required_keys(self, pipeline_and_sample):
        pipeline, sample, _ = pipeline_and_sample
        result = pipeline.predict_single(sample)
        required = {"prediction", "confidence", "tier_used", "anomaly_score"}
        assert required.issubset(result.keys())

    def test_predict_single_label_is_string(self, pipeline_and_sample):
        pipeline, sample, _ = pipeline_and_sample
        result = pipeline.predict_single(sample)
        assert isinstance(result["prediction"], str)

    def test_predict_single_confidence_in_range(self, pipeline_and_sample):
        pipeline, sample, _ = pipeline_and_sample
        result = pipeline.predict_single(sample)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch_returns_dict(self, pipeline_and_sample):
        pipeline, sample, preprocessor = pipeline_and_sample
        batch = pd.concat([sample, sample, sample], ignore_index=True)
        result = pipeline.predict_batch(batch)
        assert isinstance(result, dict)
        assert len(result["predictions"]) == 3

    def test_predict_batch_tier_flags_valid(self, pipeline_and_sample):
        pipeline, sample, _ = pipeline_and_sample
        batch = pd.concat([sample] * 5, ignore_index=True)
        result = pipeline.predict_batch(batch)
        assert all(t in {1, 2} for t in result["tier_used"])
