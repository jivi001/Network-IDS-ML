"""
Adversarial validation suite.
Tests NIDS pipeline against evasion tactics (noise injection, boundary probing).
"""

import numpy as np
import logging

logger = logging.getLogger("nids.adversarial")

class AdversarialValidator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run_suite(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Run all adversarial resilience tests."""
        report = {}
        report['noise_degradation'] = self._test_noise_injection(X_val, y_val)
        report['boundary_probing'] = self._test_boundary_evasion(X_val, y_val)
        
        # Compute total robustness score
        robustness = (1.0 - report['noise_degradation']['drop_pct']) * 
                     (1.0 - report['boundary_probing']['evasion_success_rate'])
                     
        report['overall_robustness_score'] = round(robustness, 3)
        return report

    def _test_noise_injection(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Injects Gaussian noise to simulate sensor degradation or fuzzing."""
        # Baseline
        base_preds = self.pipeline.predict(X)
        
        # Inject 5% noise
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        
        # Predict on noisy
        noisy_preds = self.pipeline.predict(X_noisy)
        
        # Compare
        match_rate = np.mean(base_preds == noisy_preds)
        
        return {
            "noise_level": 0.05,
            "prediction_stability": round(match_rate, 3),
            "drop_pct": round(1.0 - match_rate, 3)
        }

    def _test_boundary_evasion(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Simulates an attacker slightly shifting payload size/duration to evade detection."""
        # Isolate known attacks
        attack_mask = y != 'Normal'
        if not attack_mask.any():
            return {"evasion_success_rate": 0.0, "note": "No attacks to test."}
            
        X_attacks = X[attack_mask]
        
        # Apply small targeted shift (-10% to random features)
        shift = np.random.uniform(0.9, 1.0, X_attacks.shape)
        X_evasion = X_attacks * shift
        
        evasion_preds = self.pipeline.predict(X_evasion)
        
        # How many attacks successfully predicted as Normal?
        evaded = np.sum(evasion_preds == 'Normal')
        total = len(X_attacks)
        
        return {
            "attacks_tested": total,
            "successful_evasions": int(evaded),
            "evasion_success_rate": round(evaded / total, 3) if total > 0 else 0.0
        }
