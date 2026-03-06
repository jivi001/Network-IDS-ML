"""
Hybrid NIDS: 4-tier cascade architecture (upgraded).

  Tier 1 — Known-attack classifier
           Default:  SupervisedModel (BalancedRandomForestClassifier)
           Upgraded: StackingEnsemble (BRF + LightGBM + CalibratedSVC + LR meta)

  Tier 2 — Zero-day anomaly detector
           Default:  UnsupervisedModel (IsolationForest)
           Upgraded: FusionAnomalyDetector (VAE + IsolationForest fusion)

Cascade Logic:
  All traffic → Tier 1 → [High-conf Attack?] → Immediate Alert
                       → [Uncertain / Normal] → Tier 2 → [Anomaly?] → Zero-Day Alert
                                                        → [Normal?] → Pass

Backward-compatible: use_stacking=False and use_vae=False reproduce v1 behaviour.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from nids.models.supervised import SupervisedModel
from nids.models.unsupervised import UnsupervisedModel


class HybridNIDS:
    """
    Production hybrid intrusion detection system.

    Tier 1: Known-attack classifier — BalancedRF (default) or StackingEnsemble
    Tier 2: Zero-day detector      — IsolationForest (default) or FusionAnomalyDetector

    Args:
        rf_params:       Hyperparameters for SupervisedModel (Tier 1 default).
        iforest_params:  Hyperparameters for UnsupervisedModel (Tier 2 default).
        stacking_params: Hyperparameters for StackingEnsemble (Tier 1 upgraded).
        fusion_params:   Hyperparameters for FusionAnomalyDetector (Tier 2 upgraded).
        use_stacking:    If True, use StackingEnsemble as Tier 1 (slower train, higher accuracy).
        use_vae:         If True, use FusionAnomalyDetector as Tier 2 (VAE + IForest fusion).
        random_state:    Global seed for reproducibility.
    """

    def __init__(
        self,
        rf_params: Dict = None,
        iforest_params: Dict = None,
        stacking_params: Dict = None,
        fusion_params: Dict = None,
        use_stacking: bool = False,
        use_vae: bool = False,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.use_stacking = use_stacking
        self.use_vae = use_vae

        # ── Tier 1 ────────────────────────────────────────────────────────
        if use_stacking:
            from nids.models.stacking import StackingEnsemble
            stacking_params = stacking_params or {}
            stacking_params.setdefault('random_state', random_state)
            self.tier1_model = StackingEnsemble(**stacking_params)
            print("[HybridNIDS] Tier 1: StackingEnsemble (BRF + LightGBM + CalibratedSVC)")
        else:
            rf_params = rf_params or {}
            rf_params.setdefault('random_state', random_state)
            self.tier1_model = SupervisedModel(**rf_params)
            print("[HybridNIDS] Tier 1: BalancedRandomForestClassifier")

        # ── Tier 2 ────────────────────────────────────────────────────────
        if use_vae:
            from nids.models.anomaly import FusionAnomalyDetector
            fusion_params = fusion_params or {}
            fusion_params.setdefault('random_state', random_state)
            self.tier2_model = FusionAnomalyDetector(**fusion_params)
            print("[HybridNIDS] Tier 2: FusionAnomalyDetector (VAE + IsolationForest)")
        else:
            iforest_params = iforest_params or {}
            iforest_params.setdefault('random_state', random_state)
            self.tier2_model = UnsupervisedModel(**iforest_params)
            print("[HybridNIDS] Tier 2: IsolationForest")

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

        # === TIER 2: Train anomaly detector on NORMAL traffic ONLY ===
        tier2_label = "FusionAnomalyDetector" if self.use_vae else "IsolationForest"
        print("\n" + "=" * 50)
        print(f"TIER 2: Training {tier2_label}")
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

    def predict(self, X: np.ndarray, confidence_threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cascade prediction with Uncertainty Routing.
        Routes to Tier 2 if Tier 1 is uncertain OR predicts Normal.
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before prediction")

        n_samples = X.shape[0]
        final_labels = np.empty(n_samples, dtype=object)
        tier_flags = np.zeros(n_samples, dtype=int)

        # --- Tier 1: RF Classification ---
        tier1_preds = self.tier1_model.predict(X)
        tier1_proba = self.tier1_model.predict_proba(X)
        max_conf = np.max(tier1_proba, axis=1)

        # High confidence Attack bypasses Tier 2
        high_conf_attack_mask = (tier1_preds != self.normal_label) & (max_conf >= confidence_threshold)

        # Assign Tier 1 Final Outcomes
        final_labels[high_conf_attack_mask] = tier1_preds[high_conf_attack_mask]
        tier_flags[high_conf_attack_mask] = 1

        # --- Tier 2: iForest on uncertain or 'Normal' samples ---
        tier2_mask = ~high_conf_attack_mask
        if tier2_mask.sum() > 0:
            X_tier2 = X[tier2_mask]
            tier2_preds = self.tier2_model.predict(X_tier2)

            # Route 1: Tier 2 claims Anomaly
            anomaly_idx = tier2_preds == -1
            
            # Route 2: Tier 2 claims Normal
            normal_idx = tier2_preds == 1
            
            tier2_labels = np.empty(tier2_mask.sum(), dtype=object)
            
            # If Tier 2 says anomaly, it's an anomaly.
            tier2_labels[anomaly_idx] = 'Zero_Day_Anomaly'
            
            # If Tier 2 says normal, but Tier 1 thought it was an attack (with low confidence), it is suspicious
            t1_original_preds = tier1_preds[tier2_mask]
            suspicious_mask = normal_idx & (t1_original_preds != self.normal_label)
            tier2_labels[suspicious_mask] = 'Suspicious_Low_Conf_Attack'
            
            # If Tier 2 says normal and Tier 1 thought it was normal, it's normal.
            pure_normal_mask = normal_idx & (t1_original_preds == self.normal_label)
            tier2_labels[pure_normal_mask] = self.normal_label

            final_labels[tier2_mask] = tier2_labels
            tier_flags[tier2_mask] = 2

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
        if self.use_vae:
            # FusionAnomalyDetector has separate save paths for VAE + IForest
            import os
            vae_path = tier2_path.replace('.pkl', '_vae.pt')
            if_path  = tier2_path.replace('.pkl', '_iforest.pkl')
            self.tier2_model.save(vae_path, if_path)
        else:
            self.tier2_model.save(tier2_path)
        print(f"[HybridNIDS] System saved")

    def load(self, tier1_path: str, tier2_path: str, normal_label: str):
        self.tier1_model.load(tier1_path)
        if self.use_vae:
            vae_path = tier2_path.replace('.pkl', '_vae.pt')
            if_path  = tier2_path.replace('.pkl', '_iforest.pkl')
            self.tier2_model.load(vae_path, if_path)
        else:
            self.tier2_model.load(tier2_path)
        self.normal_label = normal_label
        self.is_trained = True
        print(f"[HybridNIDS] System loaded")
