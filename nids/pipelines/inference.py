"""
Inference pipeline for production predictions.
"""

from pathlib import Path
import numpy as np
import joblib

from nids.models import HybridNIDS


class InferencePipeline:
    """Production inference pipeline."""
    
    def __init__(self, model_version: str = 'v1.0.0'):
        self.model_version = model_version
        self.model_dir = Path(f'models/production/{model_version}')
        self.model = None
        self.preprocessor = None
        self.selector = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model, preprocessor, and selector."""
        # Load model
        tier1_path = self.model_dir / 'tier1_rf.pkl'
        tier2_path = self.model_dir / 'tier2_iforest.pkl'
        
        self.model = HybridNIDS()
        self.model.load(str(tier1_path), str(tier2_path), normal_label='Normal')
        
        # Load preprocessor
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
        
        # Load selector
        selector_path = self.model_dir / 'feature_selector.pkl'
        if selector_path.exists():
            self.selector = joblib.load(selector_path)
    
    def predict_single(self, features: np.ndarray) -> dict:
        """
        Predict for a single sample.
        
        Args:
            features: Feature vector (1D array)
            
        Returns:
            Dictionary with prediction and metadata
        """
        import pandas as pd
        
        # Convert to DataFrame for preprocessing
        if isinstance(features, list):
            features = pd.DataFrame([features], columns=self.preprocessor.get_feature_names())
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = pd.DataFrame(features, columns=self.preprocessor.get_feature_names())
        
        # Preprocess
        X_processed = self.preprocessor.transform(features)
        
        # Feature selection
        X_selected = self.selector.transform(X_processed)
        
        # Predict
        detailed = self.model.predict_with_scores(X_selected)
        
        return {
            'prediction': detailed['final_predictions'][0],
            'tier_used': int(detailed['tier_used'][0]),
            'anomaly_score': float(detailed['tier2_anomaly_scores'][0]),
            'confidence': float(np.max(detailed['tier1_probabilities'][0]))
        }
    
    def predict_batch(self, features: np.ndarray) -> dict:
        """
        Predict for multiple samples.
        
        Args:
            features: Feature matrix (2D array)
            
        Returns:
            Dictionary with predictions and metadata
        """
        import pandas as pd
        
        # Convert to DataFrame
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=self.preprocessor.get_feature_names())
        
        # Preprocess
        X_processed = self.preprocessor.transform(features)
        
        # Feature selection
        X_selected = self.selector.transform(X_processed)
        
        # Predict
        detailed = self.model.predict_with_scores(X_selected)
        
        return {
            'predictions': detailed['final_predictions'].tolist(),
            'tier_used': detailed['tier_used'].tolist(),
            'anomaly_scores': detailed['tier2_anomaly_scores'].tolist()
        }
