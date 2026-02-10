"""
Hybrid NIDS: Two-tier cascade architecture.
Tier 1 (Random Forest) → Known attacks
Tier 2 (Isolation Forest) → Zero-day anomalies

Cascade Logic:
  All traffic → Tier 1 → [Attack?] → Alert (immediate)
                       → [Normal?] → Tier 2 → [Anomaly?] → Zero-Day Alert
                                             → [Normal?] → Pass
"""

import numpy as np
from typing import Tuple, Dict, Optional
from nids.models.supervised import SupervisedModel
from nids.models.unsupervised import UnsupervisedModel


class HybridNIDS:
    """
    Production hybrid intrusion detection system.

    Architecture per research report:
    - Tier 1: RF trained on SMOTE-balanced, feature-selected data
    - Tier 2: iForest trained on Normal-only samples from training data
    - Cascade reduces iForest false positives by pre-filtering known attacks
    """

    def __init__(
        self,
        rf_params: Dict = None,
        iforest_params: Dict = None,
        random_state: int = 42
    ):
        self.random_state = random_state

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

        self.normal_label = None
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        normal_label: str = 'Normal'
    ):
        """
        Train both tiers.

        Args:
            X_train: Preprocessed (and optionally SMOTE-balanced) features
            y_train: Labels
            normal_label: String label for normal/benign traffic
        """
        self.normal_label = normal_label

        # === TIER 1: Train RF on ALL labeled data (balanced) ===
        print("\n" + "=" * 50)
        print("TIER 1: Training Random Forest")
        print("=" * 50)
        self.tier1_model.train(X_train, y_train)

        # === TIER 2: Train iForest on NORMAL traffic ONLY ===
        print("\n" + "=" * 50)
        print("TIER 2: Training Isolation Forest")
        print("=" * 50)

        normal_mask = (y_train == normal_label)
        X_normal = X_train[normal_mask]

        if len(X_normal) == 0:
            raise ValueError(
                f"No samples with label '{normal_label}' found. "
                f"Available labels: {np.unique(y_train)}"
            )

        print(f"  Normal samples for Tier 2: {len(X_normal)}")
        self.tier2_model.train(X_normal)

        self.is_trained = True
        print("\n[OK] Hybrid NIDS Training Complete")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cascade prediction.

        Returns:
            (final_labels, tier_flags)
            - final_labels: predicted class for each sample
            - tier_flags: 1 = decided by Tier 1, 2 = decided by Tier 2
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before prediction")

        n_samples = X.shape[0]
        final_labels = np.empty(n_samples, dtype=object)
        tier_flags = np.zeros(n_samples, dtype=int)

        # --- Tier 1: RF Classification ---
        tier1_preds = self.tier1_model.predict(X)

        # Non-normal predictions are final (attack detected)
        attack_mask = (tier1_preds != self.normal_label)
        final_labels[attack_mask] = tier1_preds[attack_mask]
        tier_flags[attack_mask] = 1

        # --- Tier 2: iForest on Tier-1-Normal samples ---
        normal_mask = ~attack_mask
        if normal_mask.sum() > 0:
            X_tier2 = X[normal_mask]
            tier2_preds = self.tier2_model.predict(X_tier2)

            # iForest: 1 = normal, -1 = anomaly
            tier2_labels = np.where(
                tier2_preds == 1,
                self.normal_label,
                'Zero_Day_Anomaly'
            )

            final_labels[normal_mask] = tier2_labels
            tier_flags[normal_mask] = 2

        return final_labels, tier_flags

    def predict_with_scores(self, X: np.ndarray) -> Dict:
        """Predict with detailed scoring information."""
        if not self.is_trained:
            raise RuntimeError("System must be trained before prediction")

        tier1_preds = self.tier1_model.predict(X)
        tier1_proba = self.tier1_model.predict_proba(X)
        tier2_scores = self.tier2_model.decision_function(X)
        final_labels, tier_flags = self.predict(X)

        return {
            'final_predictions': final_labels,
            'tier_used': tier_flags,
            'tier1_predictions': tier1_preds,
            'tier1_probabilities': tier1_proba,
            'tier1_classes': self.tier1_model.get_classes(),
            'tier2_anomaly_scores': tier2_scores
        }

    def save(self, tier1_path: str, tier2_path: str):
        self.tier1_model.save(tier1_path)
        self.tier2_model.save(tier2_path)
        print(f"[HybridNIDS] System saved")

    def load(self, tier1_path: str, tier2_path: str, normal_label: str):
        self.tier1_model.load(tier1_path)
        self.tier2_model.load(tier2_path)
        self.normal_label = normal_label
        self.is_trained = True
        print(f"[HybridNIDS] System loaded")
