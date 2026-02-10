"""
Preprocessing module for NIDS data.
Handles: Cleaning (inf/NaN), Label Encoding, StandardScaler, SMOTE.
CRITICAL: SMOTE is applied ONLY to training data to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional, List
import warnings


class NIDSPreprocessor:
    """
    Unified preprocessing pipeline for network intrusion detection data.

    Pipeline Order:
    1. Clean inf/NaN values (Median imputation)
    2. Label-encode categorical features
    3. StandardScaler (Z-score normalization)
    4. (Optional) SMOTE for class balancing

    Per the research report:
    - Label Encoding is used (not One-Hot) to avoid dimensionality explosion
    - StandardScaler is critical for Isolation Forest distance calculations
    - SMOTE generates synthetic minority samples; applied to Train set ONLY
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        # Components
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

        # State
        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit all transformations on training data.
        """
        X = X.copy()

        # Identify column types
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        self.numerical_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.feature_names = X.columns.tolist()

        # 1. Clean numerical data
        if self.numerical_cols:
            X[self.numerical_cols] = self._clean_numerical(X[self.numerical_cols])

        # 2. Fit label encoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('MISSING'))
            self.label_encoders[col] = le

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

        # Clean
        if self.numerical_cols:
            existing_num = [c for c in self.numerical_cols if c in X.columns]
            if existing_num:
                X[existing_num] = self._clean_numerical(X[existing_num])

        # Encode
        X_encoded = self._encode_categorical(X)

        # Impute + Scale
        X_imputed = self.imputer.transform(X_encoded)
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _clean_numerical(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with column max finite value, NaN handled by imputer."""
        X_clean = X_num.copy()
        for col in X_clean.columns:
            if np.isinf(X_clean[col]).any():
                max_val = X_clean[col].replace([np.inf, -np.inf], np.nan).max()
                if pd.isna(max_val):
                    max_val = 0
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], max_val)
        return X_clean

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        X_encoded = X.copy()
        for col in self.categorical_cols:
            if col not in X_encoded.columns:
                continue
            le = self.label_encoders[col]
            X_encoded[col] = X_encoded[col].astype(str).fillna('MISSING')
            X_encoded[col] = X_encoded[col].apply(
                lambda x: x if x in le.classes_ else 'MISSING'
            )
            X_encoded[col] = le.transform(X_encoded[col])
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
        CRITICAL: Only use on TRAINING data, never on test/validation.
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
            new_unique, new_counts = np.unique(y_res, return_counts=True)
            for label, count in zip(new_unique, new_counts):
                print(f"  {label}: {count}")

            return X_res, y_res

        except Exception as e:
            warnings.warn(f"SMOTE failed: {e}. Returning original data.")
            return X, y

    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order."""
        return self.feature_names
