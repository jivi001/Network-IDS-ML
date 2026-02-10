import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from typing import List, Tuple, Optional


class FeatureSelector:

    def __init__(
        self,
        n_features: int = 20,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices = []
        self.selected_feature_names = []
        self.feature_importances = {}
        self.is_fitted = False
        self.rfe_model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Fit RFE (Recursive Feature Elimination) to select top features.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Labels
            feature_names: Optional list of feature names
        """
        n_cols = X.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_cols)]

        print(f"[FeatureSelector] Starting RFE to select top {self.n_features} features...")

        # Use a lightweight RF for RFE to ensure speed
        rf = RandomForestClassifier(
            n_estimators=50,       # Reduced from 100 for speed during RFE
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Initialize RFE
        # step=0.1 means remove 10% of least important features at each iteration
        self.rfe_model = RFE(
            estimator=rf,
            n_features_to_select=self.n_features,
            step=0.1,
            verbose=1
        )
        
        self.rfe_model.fit(X, y)

        # Get selected features
        self.selected_mask = self.rfe_model.support_
        self.selected_indices = np.where(self.selected_mask)[0].tolist()
        self.selected_feature_names = [feature_names[i] for i in self.selected_indices]
        
        # Store importances of the underlying estimator (for the selected features)
        # Note: RFE estimator_ attribute is the estimator fitted on the selected features
        if hasattr(self.rfe_model.estimator_, 'feature_importances_'):
            importances = self.rfe_model.estimator_.feature_importances_
            self.feature_importances = {
                name: float(imp) 
                for name, imp in zip(self.selected_feature_names, importances)
            }
        else:
            # Fallback if no importances available
            self.feature_importances = {name: 1.0 for name in self.selected_feature_names}

        self.is_fitted = True

        print(f"[FeatureSelector] RFE Complete. Selected {len(self.selected_indices)} features.")
        print(f"  Top 5 Selected: {self.selected_feature_names[:5]}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        return self.rfe_model.transform(X)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Return sorted list of (feature_name, importance) for selected features."""
        return sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def get_selected_names(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_feature_names
