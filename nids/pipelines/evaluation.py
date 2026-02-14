"""
Evaluation pipeline for assessing trained models.
"""

from pathlib import Path
import json
import joblib
import numpy as np

from nids.data import DataLoader
from nids.models import HybridNIDS
from nids.evaluation import NIDSEvaluator
from nids.utils.logging import setup_logger


class EvaluationPipeline:
    """Evaluation pipeline for trained NIDS models."""
    
    def __init__(self, model_dir: str, output_dir: str = None):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.model_dir.parent / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('evaluation')
        
    def run(self, test_data_path: str, dataset_type: str = 'nsl_kdd'):
        """
        Run evaluation pipeline.
        
        Args:
            test_data_path: Path to test dataset
            dataset_type: Type of dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Starting evaluation pipeline")
        
        # Load model
        model = self._load_model()
        
        # Load preprocessor and selector
        preprocessor = self._load_preprocessor()
        selector = self._load_selector()
        
        # Load test data
        X_test, y_test = self._load_test_data(test_data_path, dataset_type)
        
        # Preprocess
        X_processed = preprocessor.transform(X_test)
        
        # Feature selection
        X_selected = selector.transform(X_processed)
        
        # Predict
        predictions, tier_flags = model.predict(X_selected)
        detailed = model.predict_with_scores(X_selected)
        
        # Evaluate
        evaluator = NIDSEvaluator(output_dir=str(self.output_dir))
        # Add Zero_Day_Anomaly to labels
        all_labels = sorted(set(y_test.values) | {'Zero_Day_Anomaly'})

        metrics = evaluator.evaluate(
            y_true=y_test.values,
            y_pred=predictions,
            y_proba=detailed['tier1_probabilities'],
            labels=all_labels,
            normal_label='Normal'
        )
        
        # Save results
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Evaluation complete: Recall={metrics['recall']:.4f}")
        return metrics
    
    def _load_model(self):
        """Load trained model."""
        tier1_path = self.model_dir / 'tier1_rf.pkl'
        tier2_path = self.model_dir / 'tier2_iforest.pkl'
        
        model = HybridNIDS()
        model.load(str(tier1_path), str(tier2_path), normal_label='Normal')
        return model
    
    def _load_preprocessor(self):
        """Load preprocessor."""
        path = self.model_dir / 'preprocessor.pkl'
        return joblib.load(path)
    
    def _load_selector(self):
        """Load feature selector."""
        path = self.model_dir / 'feature_selector.pkl'
        return joblib.load(path)
    
    def _load_test_data(self, test_data_path, dataset_type):
        """Load test dataset."""
        loader = DataLoader(dataset_type=dataset_type)
        df = loader.load_csv(test_data_path)
        return loader.split_features_labels(df)
