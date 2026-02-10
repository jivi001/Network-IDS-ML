"""
Data validation module for schema checking and quality assurance.
Prevents silent failures from dataset schema drift.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import ks_2samp


class DatasetValidator:
    """
    Validate dataset schema and distributions.
    Detects schema drift, missing values, and data quality issues.
    """
    
    def __init__(self, expected_schema: Optional[Dict[str, str]] = None):
        """
        Initialize validator with expected schema.
        
        Args:
            expected_schema: Dict mapping column names to expected dtypes
        """
        self.expected_schema = expected_schema or {}
    
    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Validate dataset against expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check columns
        if self.expected_schema:
            missing = set(self.expected_schema.keys()) - set(df.columns)
            if missing:
                errors.append(f"Missing columns: {missing}")
            
            extra = set(df.columns) - set(self.expected_schema.keys())
            if extra:
                errors.append(f"Unexpected columns: {extra}")
            
            # Check data types
            for col, expected_dtype in self.expected_schema.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if expected_dtype not in actual_dtype:
                        errors.append(
                            f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
                        )
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        return errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check for data quality issues.
        
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'inf_values': {},
            'duplicate_rows': 0,
            'constant_columns': []
        }
        
        # Missing values per column
        missing = df.isnull().sum()
        quality_report['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                quality_report['inf_values'][col] = int(inf_count)
        
        # Duplicate rows
        quality_report['duplicate_rows'] = int(df.duplicated().sum())
        
        # Constant columns (no variance)
        for col in df.columns:
            if df[col].nunique() == 1:
                quality_report['constant_columns'].append(col)
        
        return quality_report
    
    def detect_distribution_shift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect distribution shift using Kolmogorov-Smirnov test.
        
        Args:
            reference_df: Reference dataset (e.g., training data)
            current_df: Current dataset to compare
            alpha: Significance level for KS test
            
        Returns:
            Dictionary mapping column names to KS test results
        """
        shift_report = {}
        
        # Only test numeric columns present in both DataFrames
        common_numeric = set(reference_df.select_dtypes(include=[np.number]).columns) & \
                        set(current_df.select_dtypes(include=[np.number]).columns)
        
        for col in common_numeric:
            ref_data = reference_df[col].dropna()
            curr_data = current_df[col].dropna()
            
            if len(ref_data) > 0 and len(curr_data) > 0:
                statistic, p_value = ks_2samp(ref_data, curr_data)
                shift_report[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < alpha
                }
        
        return shift_report
