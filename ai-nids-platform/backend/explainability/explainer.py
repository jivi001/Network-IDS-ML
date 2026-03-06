import shap
import numpy as np

class SHAPExplainer:
    """
    Model Explainability module using SHAP (TreeExplainer).
    Calculates exact feature contributions for a given alert.
    """
    def __init__(self):
        self.explainer = None
        self.feature_names = [f"feature_{i}" for i in range(49)] # Mapped from config in prod

    def explain(self, model, X: np.ndarray):
        """
        Takes the Tree-based model (e.g. BRF or LightGBM from the Stacking ensemble)
        and an input vector to return sorted feature importance.
        """
        # Lazy load explainer
        if self.explainer is None:
            # In stacking, we explain the first base learner (Random Forest) for interpretability
            self.explainer = shap.TreeExplainer(model)
            
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        
        # If multiclass, get the class with the highest probability
        if isinstance(shap_values, list):
            # Sum absolute SHAP across classes for generic importance, 
            # or return specific class logic. For simplicity, we take mean.
            mean_abs = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values[0])
            
        # Map to feature names and sort
        contributions = {name: float(val) for name, val in zip(self.feature_names, mean_abs)}
        sorted_contributions = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_contributions