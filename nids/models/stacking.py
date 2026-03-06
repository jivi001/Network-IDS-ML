"""
Stacking Ensemble for NIDS Tier 1 — Known Attack Classifier.

Architecture:
  Base learners: BalancedRandomForestClassifier + LGBMClassifier + CalibratedSVC
  Meta-learner:  Multinomial Logistic Regression (passthrough=True)

Design decisions:
  - passthrough=True feeds raw features + base-learner probabilities to meta-learner
  - CalibratedClassifierCV wraps SVC so it produces probability estimates
  - BalancedRandomForestClassifier handles class imbalance natively via bootstrap
  - LightGBM uses is_unbalance=True for imbalance handling
  - All base learners use n_jobs=-1; StackingClassifier runs them in parallel via cv=5
  - Falls back gracefully to BalancedRF-only if LightGBM or sklearn SVC are unavailable

Usage:
    ensemble = StackingEnsemble(random_state=42)
    ensemble.train(X_train, y_train)
    preds = ensemble.predict(X_test)
    proba = ensemble.predict_proba(X_test)
"""

import numpy as np
import joblib
from typing import Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class StackingEnsemble:
    """
    Production stacking ensemble for Tier 1 of NIDS.

    Drop-in replacement for SupervisedModel — exposes the same interface:
        train(), predict(), predict_proba(), get_classes(),
        get_feature_importances(), save(), load()
    """

    def __init__(
        self,
        n_estimators_brf: int = 200,
        max_depth_brf: Optional[int] = 20,
        lgbm_n_estimators: int = 300,
        lgbm_num_leaves: int = 63,
        lgbm_learning_rate: float = 0.05,
        svc_C: float = 10.0,
        svc_gamma: str = 'scale',
        lgbm_enabled: bool = True,
        svc_enabled: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.is_trained = False

        brf = BalancedRandomForestClassifier(
            n_estimators=n_estimators_brf,
            max_depth=max_depth_brf,
            criterion='gini',
            sampling_strategy='all',
            replacement=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        estimators: List = [('brf', brf)]

        if lgbm_enabled and LGBM_AVAILABLE:
            lgbm = LGBMClassifier(
                n_estimators=lgbm_n_estimators,
                num_leaves=lgbm_num_leaves,
                learning_rate=lgbm_learning_rate,
                is_unbalance=True,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1,
            )
            estimators.append(('lgbm', lgbm))
        elif lgbm_enabled and not LGBM_AVAILABLE:
            print("[StackingEnsemble] LightGBM not installed — skipping. "
                  "Install with: pip install lightgbm")

        if svc_enabled:
            svc = CalibratedClassifierCV(
                SVC(kernel='rbf', C=svc_C, gamma=svc_gamma, probability=False),
                cv=3,
                n_jobs=n_jobs,
            )
            estimators.append(('svc', svc))

        meta = LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class='auto',
            solver='lbfgs',
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            passthrough=True,
            cv=cv_folds,
            n_jobs=n_jobs,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the stacking ensemble on labeled data."""
        base_names = [name for name, _ in self.model.estimators]
        print(f"\n[StackingEnsemble] Training on {X.shape[0]} samples, "
              f"{X.shape[1]} features")
        print(f"  Base learners: {base_names}")
        print(f"  Meta-learner:  LogisticRegression")
        self.model.fit(X, y)
        self.is_trained = True
        classes_str = ', '.join(str(c) for c in self.model.classes_)
        print(f"[StackingEnsemble] Training complete. Classes: {classes_str}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Introspection helpers (mirror SupervisedModel API)
    # ------------------------------------------------------------------

    def get_classes(self) -> np.ndarray:
        self._check_trained()
        return self.model.classes_

    def get_feature_importances(self) -> np.ndarray:
        """
        Return BRF feature importances from the first base learner.
        (LightGBM importances could differ; BRF is the most interpretable.)
        """
        self._check_trained()
        brf = dict(self.model.estimators_)['brf']
        return brf.feature_importances_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        self._check_trained()
        joblib.dump(self.model, filepath)
        print(f"[StackingEnsemble] Saved to {filepath}")

    def load(self, filepath: str) -> None:
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"[StackingEnsemble] Loaded from {filepath}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "StackingEnsemble must be trained before inference. Call train() first."
            )
