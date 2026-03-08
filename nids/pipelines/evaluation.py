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
    """Evaluation pipeline for trained NIDS models.

    Args:
        model_dir: Path to the **experiment root** directory produced by
                   TrainingPipeline, e.g.:
                   ``experiments/runs/unsw_nb15_baseline_20260308_102805``
                   The pipeline internally appends ``models/`` to resolve
                   artifact paths.  Do NOT pass the ``models/`` subdirectory
                   itself or you will get a double-nested path error.
        output_dir: Where to write evaluation artefacts (confusion matrix,
                    PR curve, etc.).  Defaults to ``<model_dir>/../evaluation``.
    """

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

        # FIX: Only include synthetic Tier-2 labels (Zero_Day_Anomaly,
        # Suspicious_Low_Conf_Attack) when they ACTUALLY appear in predictions.
        # Forcing zero-support labels into the report collapses macro-avg F1
        # from ~0.99 to ~0.48, which misleads automated gates and researchers.
        predicted_set = set(predictions)
        truth_set = set(y_test.values)
        synthetic_classes = {'Zero_Day_Anomaly', 'Suspicious_Low_Conf_Attack'}
        active_synthetic = predicted_set & synthetic_classes
        all_labels = sorted(truth_set | active_synthetic)

        if active_synthetic:
            self.logger.info(
                f"Tier-2 synthetic labels present in predictions: {active_synthetic}"
            )

        metrics = evaluator.evaluate(
            y_true=y_test.values,
            y_pred=predictions,
            y_proba=detailed['tier1_probabilities'],
            labels=all_labels,
            normal_label='Normal',
        )

        # Save results
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        recall_val = metrics.get('recall', float('nan'))
        self.logger.info(f"Evaluation complete: Recall={recall_val:.4f}")
        return metrics

    
    def _load_model(self):
        """Load trained model from model_dir/models/."""
        models_dir = self.model_dir / 'models'
        tier1_path = models_dir / 'tier1_rf.pkl'
        tier2_path = models_dir / 'tier2_iforest.pkl'

        for p in (tier1_path, tier2_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model artefact not found: {p}\n"
                    "Hint: pass the experiment ROOT directory to EvaluationPipeline, "
                    "not the 'models/' subdirectory."
                )

        model = HybridNIDS()
        model.load(str(tier1_path), str(tier2_path), normal_label='Normal')
        return model

    def _load_preprocessor(self):
        """Load preprocessor from model_dir/models/preprocessor.pkl."""
        path = self.model_dir / 'models' / 'preprocessor.pkl'
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {path}")
        return joblib.load(path)

    def _load_selector(self):
        """Load feature selector from model_dir/models/feature_selector.pkl."""
        path = self.model_dir / 'models' / 'feature_selector.pkl'
        if not path.exists():
            raise FileNotFoundError(f"Feature selector not found: {path}")
        return joblib.load(path)

    def _load_test_data(self, test_data_path: str, dataset_type: str):
        """Load test dataset and split features/labels."""
        loader = DataLoader(dataset_type=dataset_type)
        df = loader.load_csv(test_data_path)
        return loader.split_features_labels(df)
