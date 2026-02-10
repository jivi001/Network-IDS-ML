"""
Model wrappers for Random Forest and Isolation Forest.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV
import joblib
from typing import Optional, Dict


class SupervisedModel:
    """
    Wrapper for Random Forest classifier.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            random_state: Seed for reproducibility
            n_jobs: Parallel jobs (-1 = all cores)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced'  # Handle residual imbalance
        )
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
        """
        self.model.fit(X, y)
        self.is_trained = True
        print(f"[SupervisedModel] Trained on {X.shape[0]} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Probability array (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Return feature importances from trained model.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """
        Save model to disk.
        """
        joblib.dump(self.model, filepath)
        print(f"[SupervisedModel] Saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk.
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"[SupervisedModel] Loaded from {filepath}")


class UnsupervisedModel:
    """
    Wrapper for Isolation Forest anomaly detector.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Isolation Forest.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            n_estimators: Number of isolation trees
            random_state: Seed for reproducibility
            n_jobs: Parallel jobs
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=False
        )
        self.is_trained = False
        self.threshold = None
        
    def train(self, X: np.ndarray):
        """
        Train Isolation Forest on normal data.
        
        Args:
            X: Training features (should be NORMAL traffic only)
        """
        self.model.fit(X)
        self.is_trained = True
        
        # Calculate threshold (decision function boundary)
        scores = self.model.decision_function(X)
        self.threshold = np.percentile(scores, 10)  # 10th percentile as threshold
        
        print(f"[UnsupervisedModel] Trained on {X.shape[0]} normal samples")
        print(f"[UnsupervisedModel] Anomaly threshold: {self.threshold:.4f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Feature array
            
        Returns:
            Labels (1 = normal, -1 = anomaly)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X: Feature array
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring")
        return self.model.decision_function(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for decision_function (sklearn compatibility).
        """
        return self.decision_function(X)
    
    def save(self, filepath: str):
        """
        Save model to disk.
        """
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold
        }, filepath)
        print(f"[UnsupervisedModel] Saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk.
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.threshold = data['threshold']
        self.is_trained = True
        print(f"[UnsupervisedModel] Loaded from {filepath}")
