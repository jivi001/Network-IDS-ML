"""
Hybrid NIDS: Cascade architecture combining Random Forest and Isolation Forest.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from src.models import SupervisedModel, UnsupervisedModel
from src.preprocessing import NIDSPreprocessor


class HybridNIDS:
    """
    Two-tier hybrid intrusion detection system.
    
    Tier 1: Random Forest (Supervised) - Detects known attacks
    Tier 2: Isolation Forest (Unsupervised) - Detects zero-day anomalies
    
    Logic:
    - All traffic passes through Tier 1
    - Traffic classified as "Normal" by Tier 1 is sent to Tier 2
    - Tier 2 flags statistical anomalies as potential zero-days
    """
    
    def __init__(
        self,
        rf_params: Dict = None,
        iforest_params: Dict = None,
        random_state: int = 42
    ):
        """
        Initialize hybrid system.
        
        Args:
            rf_params: Parameters for Random Forest
            iforest_params: Parameters for Isolation Forest
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        
        # Initialize models
        rf_params = rf_params or {}
        iforest_params = iforest_params or {}
        
        self.tier1_model = SupervisedModel(
            random_state=random_state,
            **rf_params
        )
        
        self.tier2_model = UnsupervisedModel(
            random_state=random_state,
            **iforest_params
        )
        
        # State
        self.normal_label = None  # Will be set during training
        self.label_mapping = {}   # Maps encoded labels back to original
        self.is_trained = False
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        normal_label: str = 'Normal'
    ):
        """
        Train both tiers of the hybrid system.
        
        Args:
            X_train: Preprocessed training features
            y_train: Training labels (original or encoded)
            normal_label: Label representing normal/benign traffic
        """
        self.normal_label = normal_label
        
        # === TIER 1: Train Random Forest on all labeled data ===
        print("\n=== Training Tier 1: Random Forest ===")
        self.tier1_model.train(X_train, y_train)
        
        # === TIER 2: Train Isolation Forest on NORMAL traffic only ===
        print("\n=== Training Tier 2: Isolation Forest ===")
        
        # Extract only normal samples
        normal_mask = (y_train == normal_label)
        X_normal = X_train[normal_mask]
        
        if len(X_normal) == 0:
            raise ValueError(f"No samples found with label '{normal_label}'")
        
        print(f"[HybridNIDS] Training Tier 2 on {len(X_normal)} normal samples")
        self.tier2_model.train(X_normal)
        
        self.is_trained = True
        print("\n=== Hybrid NIDS Training Complete ===")
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using cascade logic.
        
        Args:
            X: Preprocessed feature array
            
        Returns:
            (final_labels, tier_flags)
            - final_labels: Array of predictions
            - tier_flags: Array indicating which tier made the decision (1 or 2)
        """
        if not self.is_trained:
            raise RuntimeError("Hybrid system must be trained before prediction")
        
        n_samples = X.shape[0]
        final_labels = np.empty(n_samples, dtype=object)
        tier_flags = np.zeros(n_samples, dtype=int)
        
        # === TIER 1: Random Forest Classification ===
        tier1_predictions = self.tier1_model.predict(X)
        
        # Identify samples classified as "Normal" by Tier 1
        normal_mask = (tier1_predictions == self.normal_label)
        
        # All non-normal predictions are final
        final_labels[~normal_mask] = tier1_predictions[~normal_mask]
        tier_flags[~normal_mask] = 1  # Tier 1 decision
        
        # === TIER 2: Isolation Forest on Tier-1-Normal samples ===
        if normal_mask.sum() > 0:
            X_tier2 = X[normal_mask]
            tier2_predictions = self.tier2_model.predict(X_tier2)
            
            # Map Isolation Forest output: 1 = normal, -1 = anomaly
            tier2_labels = np.where(
                tier2_predictions == 1,
                self.normal_label,
                'Zero_Day_Anomaly'
            )
            
            final_labels[normal_mask] = tier2_labels
            tier_flags[normal_mask] = 2  # Tier 2 decision
        
        return final_labels, tier_flags
    
    def predict_with_scores(self, X: np.ndarray) -> Dict:
        """
        Predict with additional scoring information.
        
        Args:
            X: Preprocessed feature array
            
        Returns:
            Dictionary with predictions, probabilities, and anomaly scores
        """
        if not self.is_trained:
            raise RuntimeError("Hybrid system must be trained before prediction")
        
        n_samples = X.shape[0]
        
        # Tier 1 predictions and probabilities
        tier1_predictions = self.tier1_model.predict(X)
        tier1_probabilities = self.tier1_model.predict_proba(X)
        
        # Tier 2 anomaly scores (for all samples, for analysis)
        tier2_scores = self.tier2_model.decision_function(X)
        
        # Final predictions using cascade logic
        final_labels, tier_flags = self.predict(X)
        
        return {
            'final_predictions': final_labels,
            'tier_used': tier_flags,
            'tier1_predictions': tier1_predictions,
            'tier1_probabilities': tier1_probabilities,
            'tier2_anomaly_scores': tier2_scores
        }
    
    def save(self, tier1_path: str, tier2_path: str):
        """
        Save both models.
        """
        self.tier1_model.save(tier1_path)
        self.tier2_model.save(tier2_path)
        print(f"[HybridNIDS] System saved")
    
    def load(self, tier1_path: str, tier2_path: str, normal_label: str):
        """
        Load both models.
        """
        self.tier1_model.load(tier1_path)
        self.tier2_model.load(tier2_path)
        self.normal_label = normal_label
        self.is_trained = True
        print(f"[HybridNIDS] System loaded")
