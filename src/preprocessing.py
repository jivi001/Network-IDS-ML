"""
Preprocessing module for NIDS data.
Handles cleaning, encoding, scaling, and SMOTE balancing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import warnings


class NIDSPreprocessor:
    """
    Unified preprocessing pipeline for network intrusion detection data.
    
    Pipeline:
    1. Clean inf/NaN values
    2. Encode categorical features
    3. Scale numerical features
    4. (Optional) Apply SMOTE for class balancing
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        
        # Components
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # State tracking
        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit preprocessing transformations on training data.
        
        Args:
            X: Feature DataFrame
            y: Labels (unused, for sklearn compatibility)
            
        Returns:
            self
        """
        X = X.copy()
        
        # Identify column types
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = X.columns.tolist()
        
        # Clean numerical data
        X[self.numerical_cols] = self._clean_numerical(X[self.numerical_cols])
        
        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Handle unseen categories by adding a placeholder
            le.fit(X[col].astype(str).fillna('missing'))
            self.label_encoders[col] = le
        
        # Encode categoricals
        X_encoded = self._encode_categorical(X)
        
        # Fit imputer and scaler on numerical data
        self.imputer.fit(X_encoded)
        X_imputed = self.imputer.transform(X_encoded)
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Preprocessed numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        
        # Clean numerical
        X[self.numerical_cols] = self._clean_numerical(X[self.numerical_cols])
        
        # Encode categorical
        X_encoded = self._encode_categorical(X)
        
        # Impute and scale
        X_imputed = self.imputer.transform(X_encoded)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Labels (unused)
            
        Returns:
            Preprocessed numpy array
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _clean_numerical(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """
        Replace inf with column max, handle NaN.
        
        Args:
            X_num: Numerical features only
            
        Returns:
            Cleaned DataFrame
        """
        X_clean = X_num.copy()
        
        for col in X_clean.columns:
            # Replace inf with max finite value
            if np.isinf(X_clean[col]).any():
                max_val = X_clean[col].replace([np.inf, -np.inf], np.nan).max()
                if pd.isna(max_val):
                    max_val = 0  # Fallback if all values are inf
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], max_val)
        
        return X_clean
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.
        
        Args:
            X: Full DataFrame
            
        Returns:
            DataFrame with encoded categoricals
        """
        X_encoded = X.copy()
        
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            # Handle unseen categories
            X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
            
            # Map unseen values to a default encoding
            X_encoded[col] = X_encoded[col].apply(
                lambda x: x if x in le.classes_ else 'missing'
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
        
        Args:
            X: Feature array (preprocessed)
            y: Label array
            sampling_strategy: SMOTE strategy ('auto', 'minority', or dict)
            k_neighbors: Number of neighbors for SMOTE
            
        Returns:
            (X_resampled, y_resampled)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before applying SMOTE")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        
        # Adjust k_neighbors if minority class is too small
        if min_class_count <= k_neighbors:
            k_neighbors = max(1, min_class_count - 1)
            warnings.warn(
                f"Adjusted k_neighbors to {k_neighbors} due to small minority class size"
            )
        
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"[SMOTE] Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            warnings.warn(f"SMOTE failed: {str(e)}. Returning original data.")
            return X, y
    
    def get_feature_names(self) -> list:
        """
        Return list of feature names in order.
        """
        return self.feature_names
