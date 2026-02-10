"""
SHAP-based explainability for model interpretability.
Provides feature importance and decision explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based feature importance for hybrid NIDS.
    Enables interpretability for security analysts.
    """
    
    def __init__(self):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        self.explainer = None
    
    def explain_prediction(
        self,
        model,
        X_sample: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Generate SHAP values for a single prediction.
        
        Args:
            model: Trained Random Forest model
            X_sample: Single sample (1D array or 2D with 1 row)
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to SHAP values
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For multi-class, shap_values is a list
        if isinstance(shap_values, list):
            # Use the predicted class
            pred_class = model.predict(X_sample)[0]
            class_idx = list(model.classes_).index(pred_class)
            shap_values = shap_values[class_idx]
        
        # Get SHAP values for the sample
        sample_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        return dict(zip(feature_names, sample_shap))
    
    def plot_feature_importance(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        output_path: Optional[str] = None,
        max_display: int = 20
    ):
        """
        Plot global feature importance using SHAP.
        
        Args:
            model: Trained Random Forest model
            X: Feature matrix (sample of data for SHAP computation)
            feature_names: List of feature names
            output_path: Path to save plot (optional)
            max_display: Maximum number of features to display
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For multi-class, average across classes
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[SHAP] Feature importance plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_top_features(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        """
        Get top K most important features globally.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            List of dicts with feature name and importance
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For multi-class, average across classes
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values).mean(axis=0)
        
        # Compute mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        feature_importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, mean_shap)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance[:top_k]
