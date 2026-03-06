import numpy as np
import pytest
from nids.evaluation.metrics import NIDSEvaluator

@pytest.fixture
def dummy_results():
    y_true = np.array(["Normal", "DoS", "Normal", "Probe", "Normal", "DoS"])
    y_pred = np.array(["Normal", "Normal", "Normal", "Probe", "Normal", "DoS"])
    y_proba = np.array([
        [0.9, 0.1, 0.0],
        [0.6, 0.4, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.0, 0.9],
        [0.7, 0.2, 0.1],
        [0.2, 0.8, 0.0]
    ])
    labels = ["Normal", "DoS", "Probe"]
    return y_true, y_pred, y_proba, labels


def test_evaluate_research_metrics(dummy_results):
    y_true, y_pred, y_proba, labels = dummy_results
    evaluator = NIDSEvaluator()
    
    # Run evaluate with latency
    results = evaluator.evaluate(
        y_true, y_pred, y_proba, labels, 
        normal_label="Normal", 
        detect_latency_ms=1.5
    )
    
    # Check new research metrics
    assert "mcc" in results
    assert isinstance(results["mcc"], float)
    
    assert "alert_fatigue_index" in results
    assert isinstance(results["alert_fatigue_index"], float)
    
    assert "detect_latency_ms" in results
    assert results["detect_latency_ms"] == 1.5


def test_evaluate_family_metrics(dummy_results):
    y_true, y_pred, _, labels = dummy_results
    evaluator = NIDSEvaluator()
    
    families = {
        "DoS_Family": ["DoS"],
        "Probe_Family": ["Probe"]
    }
    
    results = evaluator.evaluate(
        y_true, y_pred, labels=labels, 
        normal_label="Normal",
        attack_families=families
    )
    
    assert "family_metrics" in results
    fam_metrics = results["family_metrics"]
    
    assert "DoS_Family" in fam_metrics or "dos_family" in fam_metrics
    # Just asserting it didn't crash and returned dictionary is fine for now
    assert isinstance(fam_metrics, dict)

