"""
Feature selection module using Random Forest importance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple


class FeatureSelector:
    """
    Select top features using Random Forest feature importance.
    """
    
    def __init__(
        self, 
        n_features: int = 20,
        random_state: int = 42
    ):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of top features to select
            random_state: Seed for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = []
        self.feature_importances = {}
        self.is_fitted = False
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: List[str]
    ):
        """
        Fit Random Forest and extract feature importances.
        
        Args:
            X: Feature array
            y: Labels
            feature_names: List of feature names
            
        Returns:
            self
        """
        # Train a Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        # Get importances
        importances = rf.feature_importances_
        
        # Create importance dictionary
        self.feature_importances = {
            name: imp for name, imp in zip(feature_names, importances)
        }
        
        # Sort and select top N
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        self.selected_features = [
            name for name, _ in sorted_features[:self.n_features]
        ]
        
        self.is_fitted = True
        
        print(f"[FeatureSelector] Selected top {len(self.selected_features)} features")
        print(f"Top 5: {self.selected_features[:5]}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the top features from DataFrame.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        
        # Filter to selected features
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            print(f"[Warning] Missing features in transform: {missing}")
        
        return X[available_features]
    
    def fit_transform(
        self, 
        X_array: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        X_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X_array: Feature array for fitting RF
            y: Labels
            feature_names: List of all feature names
            X_df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        self.fit(X_array, y, feature_names)
        return self.transform(X_df)
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Return sorted list of (feature_name, importance).
        """
        return sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
