"""
Unified production model abstraction for Hybrid NIDS.

This encapsulates the entire preprocessing, feature selection,
tier-1 classification, tier-2 anomaly detection, and decision logic
into a single serializable object.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from pathlib import Path
import joblib
import json

class UnifiedHybridModel:
    """
    Unified model that wraps the entire inference pipeline:
    Preprocessor -> Feature Selector -> HybridNIDS (Tier 1 & Tier 2)
    """

    def __init__(self, preprocessor, selector, hybrid_model):
        self.preprocessor = preprocessor
        self.selector = selector
        self.hybrid_model = hybrid_model

    def validate(self, num_features=None):
        """
        Validates that the component artifacts are structurally sound
        and able to perform a safe forward pass.
        Raises ValueError or Exception if corrupted.
        """
        if not self.preprocessor or not hasattr(self.preprocessor, "transform"):
            raise ValueError("Preprocessor missing or corrupted.")
        if not self.selector or not hasattr(self.selector, "transform"):
            raise ValueError("Feature selector missing or corrupted.")
        if not self.hybrid_model or not hasattr(self.hybrid_model, "predict"):
            raise ValueError("Hybrid model missing or corrupted.")
            
        if num_features:
            dummy_x = np.zeros((1, num_features))
            try:
                self.predict(dummy_x)
            except Exception as e:
                raise RuntimeError(f"Artifact validation forward pass failed: {e}")
        return True

    def predict(self, X: Union[np.ndarray, List, pd.DataFrame]) -> Dict:
        """
        End-to-end prediction method.
        Accepts raw features and returns final predictions.
        """
        is_single = False
        if isinstance(X, list):
            if len(X) > 0 and not isinstance(X[0], (list, tuple, np.ndarray, dict)):
                is_single = True
                X = pd.DataFrame([X], columns=self.preprocessor.get_feature_names())
            else:
                X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names())
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                is_single = True
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names())
        elif isinstance(X, pd.DataFrame):
            if len(X) == 1:
                is_single = True
        elif isinstance(X, pd.Series):
            is_single = True
            X = pd.DataFrame([X.values], columns=self.preprocessor.get_feature_names())
        
        # 1. Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # 2. Extract selected features
        X_selected = self.selector.transform(X_processed)
        
        # 3. Predict via Hybrid model
        detailed = self.hybrid_model.predict_with_scores(X_selected)
        
        if is_single:
            return {
                'prediction': detailed['final_predictions'][0],
                'tier_used': int(detailed['tier_used'][0]),
                'anomaly_score': float(detailed['tier2_anomaly_scores'][0]),
                'confidence': float(np.max(detailed['tier1_probabilities'][0]))
            }
        else:
            return {
                'predictions': detailed['final_predictions'].tolist(),
                'tier_used': detailed['tier_used'].tolist(),
                'anomaly_scores': detailed['tier2_anomaly_scores'].tolist(),
                'confidence': np.max(detailed['tier1_probabilities'], axis=1).tolist()
            }

    def explain(self, X: Union[np.ndarray, List, pd.DataFrame]) -> Dict:
        """
        Explain a prediction using SHAP values.
        Accepts raw features.
        """
        from nids.explainability import SHAPExplainer
        
        if isinstance(X, list) and (len(X) == 0 or not isinstance(X[0], (list, tuple))):
            X = pd.DataFrame([X], columns=self.preprocessor.get_feature_names())
        elif isinstance(X, list):
            X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names())
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names())
            
        # Transform data to the state expected by tier 1
        X_processed = self.preprocessor.transform(X)
        X_selected = self.selector.transform(X_processed)
        
        explainer = SHAPExplainer()
        feature_names = self.selector.get_selected_names()
        
        # Evaluate SHAP on tier 1 model
        shap_values = explainer.explain_prediction(
            self.hybrid_model.tier1_model.model, X_selected, feature_names
        )
        
        # Sort by absolute SHAP value
        sorted_shap = dict(
            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return sorted_shap

    def save(self, model_dir: str):
        """Save the unified model and all its components to a directory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.preprocessor, model_dir / 'preprocessor.pkl')
        joblib.dump(self.selector, model_dir / 'feature_selector.pkl')
        
        # Save hybrid model
        self.hybrid_model.save(
            str(model_dir / 'tier1_rf.pkl'),
            str(model_dir / 'tier2_iforest.pkl')
        )
        
        # Save metadata to restore hybrid_model
        metadata = {
            'use_stacking': getattr(self.hybrid_model, 'use_stacking', False),
            'use_vae': getattr(self.hybrid_model, 'use_vae', False),
            'normal_label': getattr(self.hybrid_model, 'normal_label', 'Normal')
        }
        with open(model_dir / 'hybrid_meta.json', 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, model_dir: str):
        """Load the unified model from a directory."""
        from nids.models.hybrid import HybridNIDS
        
        model_dir = Path(model_dir)
        
        preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
        selector = joblib.load(model_dir / 'feature_selector.pkl')
        
        with open(model_dir / 'hybrid_meta.json', 'r') as f:
            metadata = json.load(f)
            
        hybrid_model = HybridNIDS(
            use_stacking=metadata.get('use_stacking', False),
            use_vae=metadata.get('use_vae', False)
        )
        hybrid_model.load(
            str(model_dir / 'tier1_rf.pkl'),
            str(model_dir / 'tier2_iforest.pkl'),
            normal_label=metadata.get('normal_label', 'Normal')
        )
        
        return cls(preprocessor, selector, hybrid_model)
