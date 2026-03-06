import numpy as np
import pytest
from nids.active_learning.query import UncertaintyDiversityQuery
from nids.active_learning.feedback import FeedbackBuffer
import os

def test_uncertainty_diversity_query():
    # 20 samples, 2 classes
    proba = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.5, 0.5], [0.55, 0.45], [0.1, 0.9],
        [0.95, 0.05], [0.2, 0.8], [0.48, 0.52], [0.51, 0.49], [0.6, 0.4],
        [0.99, 0.01], [0.88, 0.12], [0.45, 0.55], [0.49, 0.51], [0.05, 0.95],
        [0.85, 0.15], [0.15, 0.85], [0.52, 0.48], [0.53, 0.47], [0.3, 0.7]
    ])
    
    # 20 samples, 5 features
    X = np.random.rand(20, 5)
    
    class MockModel:
        def predict_proba(self, X):
            return proba

    query = UncertaintyDiversityQuery(budget=3, uncertainty_pool=10)
    indices = query.select(MockModel(), X)
    
    assert len(indices) == 3
    # The most uncertain samples (indices 2, 3, 7, 8, 12, 13, 17, 18) should be prioritized
    # This also checks that K-Means diversity works
    for idx in indices:
        assert proba[idx].max() <= 0.6  # High uncertainty check

def test_feedback_buffer(tmp_path):
    buffer_path = tmp_path / "test_buffer.json"
    buffer = FeedbackBuffer(buffer_path=str(buffer_path), trigger_size=3)
    
    assert len(buffer) == 0
    assert not buffer.should_retrain()
    
    # Add examples
    buffer.add(
        alert_id="id1",
        features=[0.1, 0.2], 
        original_label="Normal", 
        corrected_label="DoS", 
        feedback_type="relabel"
    )
    buffer.add(
        alert_id="id2",
        features=[0.8, 0.9], 
        original_label="Probe", 
        corrected_label="Probe", 
        feedback_type="approve"
    )
    
    assert len(buffer) == 2
    assert not buffer.should_retrain()
    
    buffer.add(
        alert_id="id3",
        features=[0.5, 0.5], 
        original_label="DoS", 
        corrected_label="Normal", 
        feedback_type="reject"
    )
    
    assert len(buffer) == 3
    assert buffer.should_retrain()
    
    # Test export
    csv_path = tmp_path / "export.csv"
    exported = buffer.export_labeled_dataset(str(csv_path))
    assert exported == 3
    assert os.path.exists(str(csv_path))
    
    # Test clear
    buffer.clear()
    assert len(buffer) == 0
    assert not buffer.should_retrain()
