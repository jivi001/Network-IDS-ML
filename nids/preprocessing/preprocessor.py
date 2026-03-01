"""
Preprocessing module for NIDS data.
Handles: Cleaning (inf/NaN), Frequency Encoding, StandardScaler, SMOTE.
CRITICAL: SMOTE is applied ONLY to training data to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional, List, Dict
import warnings


class NIDSPreprocessor:
    """
    Unified preprocessing pipeline for network intrusion detection data.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.missing_token = 'MISSING'

        # Components
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

        # Stateful Caches
        self.max_values_: Dict[str, float] = {}
        self.freq_encoders_: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit all transformations on training data.
        """
        X = X.copy()
        self.feature_names = X.columns.tolist()

        # Identify column types
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        self.numerical_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # 1. Clean numerical data (Learn Max Values)
        if self.numerical_cols:
            for col in self.numerical_cols:
                # Replace inf with nan temporarily to find the real max
                finite_max = X[col].replace([np.inf, -np.inf], np.nan).max()
                self.max_values_[col] = finite_max if pd.notna(finite_max) else 0.0
            X[self.numerical_cols] = self._clean_numerical(X[self.numerical_cols])

        # 2. Learn Frequency Encoders for categoricals
        for col in self.categorical_cols:
            X[col] = X[col].fillna(self.missing_token).astype(str)
            freq = X[col].value_counts(normalize=True).to_dict()
            self.freq_encoders_[col] = freq

        # Encode categoricals for fitting imputer/scaler
        X_encoded = self._encode_categorical(X)

        # 3. Fit imputer then scaler
        self.imputer.fit(X_encoded)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_encoded),
            columns=X_encoded.columns
        )
        self.scaler.fit(X_imputed)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        Returns numpy array (n_samples, n_features).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        X = X.copy()
        self.validate_shape(X)

        # Clean using stored max values
        if self.numerical_cols:
            existing_num = [c for c in self.numerical_cols if c in X.columns]
            if existing_num:
                X[existing_num] = self._clean_numerical(X[existing_num])

        # Encode using stored frequency encoders
        X_encoded = self._encode_categorical(X)

        # Impute + Scale
        X_imputed = self.imputer.transform(X_encoded)
        # Cast to float32 for optimization
        X_scaled = self.scaler.transform(X_imputed).astype(np.float32)

        self.validate_no_nan(X_scaled)
        return X_scaled

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _clean_numerical(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with stored max finite values."""
        X_clean = X_num.copy()
        for col in X_clean.columns:
            if np.isinf(X_clean[col]).any():
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], self.max_values_[col])
        return X_clean

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding to categorical columns."""
        X_encoded = X.copy()
        for col in self.categorical_cols:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].fillna(self.missing_token).astype(str)
                # Map frequency, default to 0.0001 (rare) for unseen
                X_encoded[col] = X_encoded[col].map(self.freq_encoders_[col]).fillna(0.0001)
        return X_encoded

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: str = 'auto',
        k_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance classes.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before applying SMOTE")

        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        # Adjust k_neighbors for small classes
        if min_count <= k_neighbors:
            k_neighbors = max(1, min_count - 1)
            warnings.warn(
                f"Adjusted SMOTE k_neighbors to {k_neighbors} (minority class has {min_count} samples)"
            )

        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
            X_res, y_res = smote.fit_resample(X, y)

            print(f"[SMOTE] {X.shape[0]} -> {X_res.shape[0]} samples")
            return X_res, y_res

        except Exception as e:
            warnings.warn(f"SMOTE failed: {e}. Returning original data.")
            return X, y

    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order."""
        return self.feature_names

    def validate_shape(self, X: pd.DataFrame):
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Feature shape mismatch. Missing: {missing_cols}")

    def validate_no_nan(self, X: np.ndarray):
        if np.isnan(X).any():
            raise ValueError("CRITICAL: NaN values propagated past the imputer.")
