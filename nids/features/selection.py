import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from typing import List, Tuple, Optional


class FeatureSelector:

    def __init__(
        self,
        n_features: int = 20,
        method: str = 'rfe',
        random_state: int = 42
    ):
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        self.selected_indices = []
        self.selected_feature_names = []
        self.feature_importances = {}
        self.is_fitted = False
        self.selector_model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Fit feature selector.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Labels
            feature_names: Optional list of feature names
        """
        n_cols = X.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_cols)]

        print(f"[FeatureSelector] Starting selection ({self.method}) to select top {self.n_features} features...")

        # Use a lightweight RF
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        if self.method == 'importance':
            from sklearn.feature_selection import SelectFromModel
            # SelectFromModel doesn't allow exact n_features easily, but we can set threshold
            # or just start with it. For strict n_features, we might need manual selection.
            # Let's use manual selection based on importance to get exactly n_features
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:self.n_features]
            
            self.selected_mask = np.zeros(n_cols, dtype=bool)
            self.selected_mask[indices] = True
            self.selector_model = rf  # Store the fitted model
            
        else:
            # Default to RFE
            self.selector_model = RFE(
                estimator=rf,
                n_features_to_select=self.n_features,
                step=0.1,
                verbose=1
            )
            self.selector_model.fit(X, y)
            self.selected_mask = self.selector_model.support_

        # Get selected features
        self.selected_indices = np.where(self.selected_mask)[0].tolist()
        self.selected_feature_names = [feature_names[i] for i in self.selected_indices]
        
        # Store importances
        if self.method == 'importance':
            importances = self.selector_model.feature_importances_
            # Map all importances
            self.feature_importances = {
                name: float(importances[i])
                for i, name in enumerate(feature_names)
                if i in self.selected_indices
            }
        else:
            # RFE case
            if hasattr(self.selector_model.estimator_, 'feature_importances_'):
                importances = self.selector_model.estimator_.feature_importances_
                self.feature_importances = {
                    name: float(imp) 
                    for name, imp in zip(self.selected_feature_names, importances)
                }
            else:
                self.feature_importances = {name: 1.0 for name in self.selected_feature_names}

        self.is_fitted = True

        print(f"[FeatureSelector] RFE Complete. Selected {len(self.selected_indices)} features.")
        print(f"  Top 5 Selected: {self.selected_feature_names[:5]}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_mask]

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
