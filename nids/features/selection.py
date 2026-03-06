"""
Feature Selection for NIDS — upgraded with SHAP, Mutual Information, and Borda count.

Methods supported:
  'importance'    — RF Gini importance (original, fast)
  'rfe'           — Recursive Feature Elimination (original, thorough)
  'shap'          — SHAP TreeExplainer mean |SHAP| values (most reliable)
  'mutual_info'   — Mutual Information between features and labels
  'combined'      — Borda count rank fusion: RF importance + SHAP + MI (best overall)

All methods expose the same interface:
    fit(X, y, feature_names) → self
    transform(X) → np.ndarray
    fit_transform(X, y, feature_names) → np.ndarray
    get_selected_names() → List[str]
    get_feature_importance_ranking() → List[Tuple[str, float]]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from typing import List, Tuple, Optional


class FeatureSelector:
    """
    Unified feature selector with SHAP, Mutual Information, and Borda count fusion.

    Args:
        n_features (int): Number of features to select.
        method (str): One of 'importance', 'rfe', 'shap', 'mutual_info', 'combined'.
        random_state (int): Random seed.
        shap_sample_size (int): Max samples used for SHAP computation (for speed).
    """

    VALID_METHODS = {'importance', 'rfe', 'shap', 'mutual_info', 'combined'}

    def __init__(
        self,
        n_features: int = 20,
        method: str = 'importance',
        random_state: int = 42,
        shap_sample_size: int = 2000,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Valid: {sorted(self.VALID_METHODS)}"
            )
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        self.shap_sample_size = shap_sample_size

        self.selected_indices: List[int] = []
        self.selected_feature_names: List[str] = []
        self.feature_importances: dict = {}
        self.selected_mask: Optional[np.ndarray] = None
        self.selector_model = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'FeatureSelector':
        """Fit selector and compute selected feature mask."""
        n_cols = X.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_cols)]

        n_select = min(self.n_features, n_cols)
        print(f"[FeatureSelector] method='{self.method}', "
              f"selecting top {n_select}/{n_cols} features...")

        if self.method == 'importance':
            self.selected_mask = self._rf_importance(X, y, n_select)

        elif self.method == 'rfe':
            self.selected_mask = self._rfe(X, y, n_select)

        elif self.method == 'shap':
            self.selected_mask = self._shap_importance(X, y, n_select)

        elif self.method == 'mutual_info':
            self.selected_mask = self._mutual_info(X, y, n_select)

        elif self.method == 'combined':
            self.selected_mask = self._borda_fusion(X, y, n_select)

        # Resolve names and scores
        self.selected_indices = np.where(self.selected_mask)[0].tolist()
        self.selected_feature_names = [feature_names[i] for i in self.selected_indices]

        # Store importance values for the selected features
        if hasattr(self, '_scores') and self._scores is not None:
            self.feature_importances = {
                feature_names[i]: float(self._scores[i])
                for i in self.selected_indices
            }
        else:
            self.feature_importances = {
                name: 1.0 for name in self.selected_feature_names
            }

        self.is_fitted = True
        print(f"[FeatureSelector] Done. Selected {len(self.selected_indices)} features.")
        print(f"  Top 5: {self.selected_feature_names[:5]}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform.")
        return X[:, self.selected_mask]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_names(self) -> List[str]:
        return self.selected_feature_names

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Return sorted list of (feature_name, importance) for selected features."""
        return sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _rf_importance(self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """Original RF Gini importance method."""
        rf = self._make_light_rf()
        rf.fit(X, y)
        self.selector_model = rf
        importances = rf.feature_importances_
        self._scores = importances
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(importances)[::-1][:n]] = True
        return mask

    def _rfe(self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """Original RFE method."""
        rf = self._make_light_rf()
        rfe = RFE(estimator=rf, n_features_to_select=n, step=0.1, verbose=0)
        rfe.fit(X, y)
        self.selector_model = rfe
        # Build importances from the fitted estimator
        if hasattr(rfe.estimator_, 'feature_importances_'):
            raw = rfe.estimator_.feature_importances_
            self._scores = np.zeros(X.shape[1])
            selected_idx = np.where(rfe.support_)[0]
            for k, idx in enumerate(selected_idx):
                if k < len(raw):
                    self._scores[idx] = raw[k]
        else:
            self._scores = None
        return rfe.support_

    def _shap_importance(self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """SHAP TreeExplainer mean |SHAP| ranking."""
        try:
            import shap
        except ImportError:
            print("[FeatureSelector] shap not installed, falling back to 'importance'.")
            return self._rf_importance(X, y, n)

        rf = self._make_light_rf()
        rf.fit(X, y)
        self.selector_model = rf

        # Sample for speed
        n_sample = min(self.shap_sample_size, X.shape[0])
        rng = np.random.default_rng(self.random_state)
        sample_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
        X_sample = X[sample_idx]

        explainer = shap.TreeExplainer(rf, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # shap_values: list of (n_sample, n_features) per class, or 2D for binary
        if isinstance(shap_values, list):
            # Multi-class: average |SHAP| across all classes
            mean_abs = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
            if len(mean_abs.shape) > 1:
                mean_abs = mean_abs.mean(axis=1)

        self._scores = mean_abs
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(mean_abs)[::-1][:n]] = True
        print(f"  [SHAP] using {n_sample} samples for SHAP computation")
        return mask

    def _mutual_info(self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """Mutual Information feature ranking."""
        mi_scores = mutual_info_classif(
            X, y, discrete_features=False, random_state=self.random_state
        )
        self._scores = mi_scores
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(mi_scores)[::-1][:n]] = True
        return mask

    def _borda_fusion(self, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """
        Borda count rank fusion across 3 rankers:
            1. RF Gini importance
            2. SHAP mean |SHAP| (falls back to RF importance if shap unavailable)
            3. Mutual Information

        Each ranker ranks features 0..n_cols-1 (0 = least important).
        Borda score = sum of ranks. Top-N selected.
        """
        n_cols = X.shape[1]

        # --- Ranker 1: RF importance ---
        rf = self._make_light_rf()
        rf.fit(X, y)
        self.selector_model = rf
        rf_imp = rf.feature_importances_
        rf_ranks = np.argsort(np.argsort(rf_imp))  # rank 0=worst, n_cols-1=best

        # --- Ranker 2: SHAP ---
        try:
            import shap
            n_sample = min(self.shap_sample_size, X.shape[0])
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
            X_sample = X[sample_idx]
            explainer = shap.TreeExplainer(rf, feature_perturbation='interventional')
            shap_vals = explainer.shap_values(X_sample, check_additivity=False)
            if isinstance(shap_vals, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            else:
                mean_abs = np.abs(shap_vals).mean(axis=0)
                # If shap_vals was (n_samples, n_features, n_classes), mean_abs is (n_features, n_classes)
                if len(mean_abs.shape) > 1:
                    mean_abs = mean_abs.mean(axis=1)
            shap_ranks = np.argsort(np.argsort(mean_abs))
            print(f"  [Borda] SHAP computed on {n_sample} samples")
        except ImportError:
            print("  [Borda] shap not available — using RF importance for SHAP rank")
            shap_ranks = rf_ranks

        # --- Ranker 3: Mutual Information ---
        mi_scores = mutual_info_classif(
            X, y, discrete_features=False, random_state=self.random_state
        )
        mi_ranks = np.argsort(np.argsort(mi_scores))

        # --- Borda fusion: sum ranks ---
        borda_scores = rf_ranks.astype(float) + shap_ranks + mi_ranks
        self._scores = borda_scores

        mask = np.zeros(n_cols, dtype=bool)
        mask[np.argsort(borda_scores)[::-1][:n]] = True
        print(f"  [Borda] Combined RF + SHAP + MI rankings into Borda score")
        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_light_rf(self) -> RandomForestClassifier:
        """Lightweight RF for quick importance estimation."""
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced',
        )
