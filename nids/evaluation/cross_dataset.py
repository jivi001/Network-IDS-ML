"""
Cross-dataset evaluation for testing model generalization.
Evaluates models trained on one dataset using a different dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import joblib


class CrossDatasetEvaluator:
    """
    Evaluate model trained on Dataset A using Dataset B.
    Handles feature alignment and label mapping.
    """
    
    def __init__(self, model_path: str, source_dataset: str):
        """
        Initialize cross-dataset evaluator.
        
        Args:
            model_path: Path to trained model directory
            source_dataset: Name of source dataset (e.g., 'nsl_kdd')
        """
        self.model_path = Path(model_path)
        self.source_dataset = source_dataset
        self.model = None
        self.preprocessor = None
        self.feature_selector = None
        
    def load_model_artifacts(self):
        """Load trained model, preprocessor, and feature selector."""
        # Load models
        tier1_path = self.model_path / 'tier1_rf.pkl'
        tier2_path = self.model_path / 'tier2_iforest.pkl'
        
        if not tier1_path.exists():
            raise FileNotFoundError(f"Model not found: {tier1_path}")
        
        from nids.models import HybridNIDS
        self.model = HybridNIDS()
        self.model.load(str(tier1_path), str(tier2_path), normal_label='Normal')
        
        # Load preprocessor
        preprocessor_path = self.model_path / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
        
        # Load feature selector
        selector_path = self.model_path / 'feature_selector.pkl'
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
    
    def align_features(
        self,
        source_features: List[str],
        target_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Align features between source and target datasets.
        
        Args:
            source_features: Feature names from source dataset
            target_df: Target dataset DataFrame
            
        Returns:
            Tuple of (aligned_df, alignment_report)
        """
        target_features = set(target_df.columns)
        source_features_set = set(source_features)
        
        # Find common features
        common_features = source_features_set & target_features
        missing_in_target = source_features_set - target_features
        extra_in_target = target_features - source_features_set
        
        alignment_report = {
            'source_features': len(source_features),
            'target_features': len(target_features),
            'common_features': len(common_features),
            'missing_in_target': list(missing_in_target),
            'extra_in_target': list(extra_in_target)
        }
        
        # Create aligned DataFrame
        aligned_df = pd.DataFrame()
        
        for feature in source_features:
            if feature in target_df.columns:
                aligned_df[feature] = target_df[feature]
            else:
                # Fill missing features with zeros
                aligned_df[feature] = 0
                print(f"[Warning] Feature '{feature}' missing in target, filled with 0")
        
        return aligned_df, alignment_report
    
    def map_labels(
        self,
        target_labels: pd.Series,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Map target dataset labels to source dataset taxonomy.
        
        Args:
            target_labels: Labels from target dataset
            label_mapping: Optional custom mapping dict
            
        Returns:
            Mapped labels
        """
        if label_mapping is None:
            # Default mapping (identity mapping)
            label_mapping = {label: label for label in target_labels.unique()}
        
        mapped_labels = target_labels.map(label_mapping)
        
        # Handle unmapped labels
        unmapped = mapped_labels.isnull()
        if unmapped.any():
            print(f"[Warning] {unmapped.sum()} labels could not be mapped")
            mapped_labels[unmapped] = 'Unknown'
        
        return mapped_labels
    
    def evaluate_on_target(
        self,
        target_dataset: str,
        target_data_path: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate cross-dataset generalization.
        
        Args:
            target_dataset: Name of target dataset
            target_data_path: Path to target dataset CSV
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model artifacts
        self.load_model_artifacts()
        
        # Load target dataset
        from nids.data import DataLoader
        loader = DataLoader(dataset_type=target_dataset)
        target_df = loader.load_csv(target_data_path)
        X_target, y_target = loader.split_features_labels(target_df)
        
        # Align features
        if self.preprocessor:
            source_features = self.preprocessor.get_feature_names()
            X_aligned, alignment_report = self.align_features(source_features, X_target)
        else:
            X_aligned = X_target
            alignment_report = {}
        
        # Preprocess
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X_aligned)
        else:
            X_processed = X_aligned.values
        
        # Feature selection
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_processed)
        else:
            X_selected = X_processed
        
        # Predict
        predictions, tier_flags = self.model.predict(X_selected)
        
        # Evaluate
        from nids.evaluation import NIDSEvaluator
        evaluator = NIDSEvaluator(output_dir=output_dir or 'experiments/cross_dataset')
        
        metrics = evaluator.evaluate(
            y_true=y_target.values,
            y_pred=predictions,
            y_proba=None,
            labels=sorted(set(y_target.values)),
            normal_label='Normal'
        )
        
        # Create comprehensive report
        report = {
            'source_dataset': self.source_dataset,
            'target_dataset': target_dataset,
            'alignment': alignment_report,
            'metrics': metrics,
            'tier_statistics': {
                'tier1_count': int(np.sum(tier_flags == 1)),
                'tier2_count': int(np.sum(tier_flags == 2))
            }
        }
        
        # Save report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'cross_dataset_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            with open(output_path / 'feature_alignment.json', 'w') as f:
                json.dump(alignment_report, f, indent=2)
            
            print(f"[CrossDataset] Report saved to {output_dir}")
        
        return report
