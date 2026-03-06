import numpy as np
import pytest
from nids.features.selection import FeatureSelector

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    # 100 samples, 10 features
    X = np.random.rand(100, 10)
    # Make feature 0 and 1 perfectly predictive of y
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    # Add some noise to make it realistic
    y = np.where(np.random.rand(100) < 0.1, 1 - y, y)
    feature_names = [f"f{i}" for i in range(10)]
    return X, y, feature_names

def test_feature_selector_shap(dummy_data):
    X, y, feature_names = dummy_data
    selector = FeatureSelector(n_features=3, method='shap')
    
    try:
        import shap
    except ImportError:
        pytest.skip("shap is not installed")
        
    X_sel = selector.fit_transform(X, y)
    assert X_sel.shape == (100, 3)
    assert len(selector.get_selected_names()) == 3
    # Ensure predictive features are selected
    assert "f0" in selector.get_selected_names() or "f1" in selector.get_selected_names()


def test_feature_selector_mutual_info(dummy_data):
    X, y, feature_names = dummy_data
    selector = FeatureSelector(n_features=4, method='mutual_info')
    
    X_sel = selector.fit_transform(X, y)
    assert X_sel.shape == (100, 4)
    assert len(selector.get_selected_names()) == 4
    # Ensure predictive features are selected
    assert "f0" in selector.get_selected_names()
    assert "f1" in selector.get_selected_names()


def test_feature_selector_combined(dummy_data):
    X, y, feature_names = dummy_data
    selector = FeatureSelector(n_features=5, method='combined')
    
    X_sel = selector.fit_transform(X, y)
    assert X_sel.shape == (100, 5)
    assert len(selector.get_selected_names()) == 5
    assert "f0" in selector.get_selected_names()
    assert "f1" in selector.get_selected_names()


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        FeatureSelector(method="invalid_method")
