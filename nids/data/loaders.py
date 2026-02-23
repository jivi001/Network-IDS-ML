"""
Data loading module for NIDS datasets.
Supports NSL-KDD, UNSW-NB15, CIC-IDS2017 formats.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os


class DataLoader:
    """
    Handles loading and basic validation of network intrusion datasets.
    Supports chunked reading for large files.
    """
    
    # Dataset-specific column mappings
    DATASET_SCHEMAS = {
        'nsl-kdd': {
            'label_col': 'label',
            'expected_features': 41
        },
        'nsl_kdd': {
            'label_col': 'label',
            'expected_features': 41
        },
        'unsw-nb15': {
            'label_col': 'label',
            'expected_features': 49
        },
        'unsw_nb15': {
            'label_col': 'label',
            'expected_features': 49
        },
        'cic-ids2017': {
            'label_col': 'Label',
            'expected_features': 80
        },
        'cic_ids2017': {
            'label_col': 'Label',
            'expected_features': 80
        }
    }
    
    def __init__(self, dataset_type: str = 'auto'):
        """
        Initialize DataLoader.
        
        Args:
            dataset_type: One of ['nsl-kdd', 'unsw-nb15', 'cic-ids2017', 'auto']
        """
        self.dataset_type = dataset_type
        self.label_col = None
        
    def load_csv(
        self, 
        filepath: str, 
        chunksize: Optional[int] = None,
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load CSV file with optional chunking.
        
        Args:
            filepath: Path to CSV file
            chunksize: If specified, return iterator for chunked reading
            nrows: Limit number of rows (for testing)
            
        Returns:
            DataFrame or iterator of DataFrames
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        # Attempt to load with automatic inference
        try:
            df = pd.read_csv(
                filepath, 
                chunksize=chunksize,
                nrows=nrows,
                low_memory=False
            )
            
            if chunksize is None:
                # Single DataFrame - detect schema
                self._detect_schema(df)
                return self._validate_dataframe(df)
            else:
                # Return iterator (schema detection deferred)
                return df
                
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    def _detect_schema(self, df: pd.DataFrame) -> None:
        """
        Auto-detect dataset type based on columns and shape.
        """
        if self.dataset_type != 'auto':
            schema = self.DATASET_SCHEMAS.get(self.dataset_type)
            if schema:
                self.label_col = schema['label_col']
                return
        
        # Auto-detection logic
        if 'Label' in df.columns:
            self.dataset_type = 'cic-ids2017'
            self.label_col = 'Label'
        elif 'label' in df.columns:
            # Distinguish between NSL-KDD and UNSW-NB15 by feature count
            if len(df.columns) <= 45:
                self.dataset_type = 'nsl-kdd'
            else:
                self.dataset_type = 'unsw-nb15'
            self.label_col = 'label'
        else:
            # Fallback: assume last column is label
            self.label_col = df.columns[-1]
            self.dataset_type = 'unknown'
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic validation and cleaning.
        """
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataset")
        
        # Normalize binary numeric labels (0/1) to string labels for UNSW-NB15
        if df[self.label_col].dtype in [np.int64, np.float64, int, float]:
            df[self.label_col] = df[self.label_col].astype(int).map(
                {0: 'Normal', 1: 'Attack'}
            ).fillna('Normal')
            print(f"[DataLoader] Mapped numeric labels to Normal/Attack")
        
        # Remove duplicate rows
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        
        if df.shape[0] < initial_shape:
            print(f"[DataLoader] Removed {initial_shape - df.shape[0]} duplicate rows")
        
        return df
    
    def split_features_labels(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and labels (y).
        
        Args:
            df: Full DataFrame
            
        Returns:
            (X, y) tuple
        """
        if self.label_col is None:
            self._detect_schema(df)
        
        y = df[self.label_col].copy()
        X = df.drop(columns=[self.label_col])
        
        return X, y
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Return summary statistics about the dataset.
        """
        X, y = self.split_features_labels(df)
        
        return {
            'dataset_type': self.dataset_type,
            'total_samples': len(df),
            'num_features': X.shape[1],
            'label_column': self.label_col,
            'class_distribution': y.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'inf_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        }
