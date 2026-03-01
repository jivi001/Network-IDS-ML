"""
Training pipeline for Hybrid NIDS.
End-to-end workflow from data loading to model training and evaluation.
Supports optional MLflow experiment tracking.
"""

from pathlib import Path
from datetime import datetime
import yaml
import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from nids.data import DataLoader
from nids.preprocessing import NIDSPreprocessor
from nids.features import FeatureSelector
from nids.models import HybridNIDS
from nids.evaluation import NIDSEvaluator
from nids.utils.config import load_config
from nids.utils.logging import setup_logger

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TrainingPipeline:
    """End-to-end training pipeline for Hybrid NIDS."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.experiment_id = self._generate_experiment_id()
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'experiments/runs')) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('training', log_file=str(self.output_dir / 'training.log'))
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config.get('experiment_name', 'exp')
        return f"{name}_{timestamp}"
    
    def run(self):
        """Execute full training pipeline."""
        self.logger.info(f"Starting experiment: {self.experiment_id}")

        # Initialize MLflow tracking if configured
        use_mlflow = self.config.get('tracking', {}).get('use_mlflow', False)
        if use_mlflow and MLFLOW_AVAILABLE:
            tracking_uri = self.config.get('tracking', {}).get('mlflow_uri', 'mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            exp_name = self.config.get('tracking', {}).get(
                'experiment_name', 'nids-hybrid'
            )
            mlflow.set_experiment(exp_name)
            self.logger.info(f"MLflow tracking enabled: {exp_name}")
        elif use_mlflow and not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow requested but not installed. pip install mlflow")
            use_mlflow = False

        # Wrap run() body in an optional MLflow run context
        if use_mlflow:
            with mlflow.start_run(run_name=self.experiment_id):
                return self._run_pipeline(use_mlflow=True)
        else:
            return self._run_pipeline(use_mlflow=False)

    def _run_pipeline(self, use_mlflow: bool = False):
        """Core pipeline logic, optionally under an active MLflow run."""
        # 1. Load data
        X_train, y_train, X_test, y_test = self._load_data()
        
        # 2. Preprocess
        X_train_processed, X_test_processed = self._preprocess(X_train, X_test)
        
        # 3. Apply SMOTE
        X_train_balanced, y_train_balanced = self._apply_smote(
            X_train_processed, y_train
        )
        
        # 4. Feature selection
        X_train_selected, X_test_selected = self._select_features(
            X_train_balanced, y_train_balanced, X_test_processed
        )
        
        # 5. Train models
        model = self._train_models(X_train_selected, y_train_balanced)
        
        # 6. Evaluate
        metrics = self._evaluate(model, X_test_selected, y_test)
        
        # 7. Save artifacts
        self._save_artifacts(model, metrics)

        # 8. Log to MLflow
        if use_mlflow:
            self._log_to_mlflow(metrics)
        
        self.logger.info(f"Experiment complete: {self.experiment_id}")
        return self.experiment_id, metrics
    
    def _load_data(self):
        """Load and split dataset."""
        dataset_config = self.config['dataset']
        
        # Load dataset config
        dataset_yaml = load_config(dataset_config['config'])
        
        loader = DataLoader(dataset_type=dataset_config['name'])
        
        if 'test' in dataset_yaml['paths']:
            # Use provided train/test split
            df_train = loader.load_csv(dataset_yaml['paths']['train'])
            df_test = loader.load_csv(dataset_yaml['paths']['test'])
            X_train, y_train = loader.split_features_labels(df_train)
            X_test, y_test = loader.split_features_labels(df_test)
        else:
            # Create train/test split
            df = loader.load_csv(dataset_yaml['paths']['train'])
            X, y = loader.split_features_labels(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=dataset_config.get('test_size', 0.3),
                random_state=dataset_config.get('random_state', 42),
                stratify=y if dataset_config.get('stratify', True) else None
            )
        
        self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def _preprocess(self, X_train, X_test):
        """Apply preprocessing pipeline."""
        preprocessor = NIDSPreprocessor(
            random_state=self.config['dataset'].get('random_state', 42)
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        self.preprocessor = preprocessor
        self.logger.info("Preprocessing complete")
        return X_train_processed, X_test_processed
    
    def _apply_smote(self, X_train, y_train):
        """Apply SMOTE if configured."""
        if not self.config['preprocessing'].get('apply_smote', True):
            return X_train, y_train
        
        X_balanced, y_balanced = self.preprocessor.apply_smote(
            X_train, y_train,
            sampling_strategy=self.config['preprocessing'].get('smote_strategy', 'auto'),
            k_neighbors=self.config['preprocessing'].get('smote_k_neighbors', 5)
        )
        self.logger.info(f"SMOTE: {X_train.shape[0]} -> {X_balanced.shape[0]}")
        return X_balanced, y_balanced
    
    def _select_features(self, X_train, y_train, X_test):
        """Apply feature selection."""
        feature_config = self.config['feature_selection']
        selector = FeatureSelector(
            n_features=feature_config.get('n_features', 20),
            method=feature_config.get('method', 'rfe'),
            random_state=self.config['dataset'].get('random_state', 42)
        )
        
        feature_names = self.preprocessor.get_feature_names()
        X_train_selected = selector.fit_transform(X_train, y_train, feature_names)
        X_test_selected = selector.transform(X_test)
        
        self.selector = selector
        self.logger.info(f"Feature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
        return X_train_selected, X_test_selected
    
    def _train_models(self, X_train, y_train):
        """Train Hybrid NIDS with optional cross-validation."""
        # Load model configs
        tier1_config = load_config(self.config['models']['tier1']['config'])
        tier2_config = load_config(self.config['models']['tier2']['config'])

        # --- Optional cross-validation (Tier 1 only, fast diagnostic) ---
        if self.config.get('training', {}).get('cross_validate', False):
            n_folds = self.config['training'].get('cv_folds', 5)
            self.logger.info(f"Running {n_folds}-fold stratified cross-validation...")
            rf_params = {**tier1_config['hyperparameters']}
            rf_params.setdefault('random_state', self.config['dataset'].get('random_state', 42))
            cv_model = RandomForestClassifier(**rf_params)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=self.config['dataset'].get('random_state', 42))
            cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv,
                                        scoring='recall_weighted', n_jobs=-1)
            self.logger.info(
                f"CV Recall (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
            self.cv_scores = cv_scores.tolist()
        else:
            self.cv_scores = None

        model = HybridNIDS(
            rf_params=tier1_config['hyperparameters'],
            iforest_params=tier2_config['hyperparameters'],
            random_state=self.config['dataset'].get('random_state', 42)
        )
        
        model.train(
            X_train, y_train,
            normal_label=self.config['training'].get('normal_label', 'Normal')
        )

        self.logger.info("Model training complete")
        return model
    
    def _evaluate(self, model, X_test, y_test):
        """Evaluate trained model."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        evaluator = NIDSEvaluator(output_dir=str(plots_dir))
        
        predictions, tier_flags = model.predict(X_test)
        detailed = model.predict_with_scores(X_test)
        
        # Add Zero_Day_Anomaly to labels if it might be predicted
        all_labels = sorted(set(y_test) | {'Zero_Day_Anomaly'})
        
        metrics = evaluator.evaluate(
            y_true=y_test,
            y_pred=predictions,
            y_proba=detailed['tier1_probabilities'],
            labels=all_labels,
            normal_label=self.config['training'].get('normal_label', 'Normal')
        )
        
        # SHAP explainability (if configured)
        if self.config['evaluation'].get('compute_shap', False):
            try:
                from nids.explainability import SHAPExplainer
                explainer = SHAPExplainer()
                explainer.plot_feature_importance(
                    model.tier1_model.model,
                    X_test[:1000],  # Sample for speed
                    self.selector.get_selected_names(),
                    output_path=str(plots_dir / 'shap_summary.png')
                )
            except Exception as e:
                self.logger.warning(f"SHAP computation failed: {e}")
        
        self.logger.info(f"Evaluation complete: Recall={metrics['recall']:.4f}")
        return metrics
    
    def _save_artifacts(self, model, metrics):
        """Save all experiment artifacts."""
        import joblib
        
        # Save models
        if self.config['output'].get('save_models', True):
            models_dir = self.output_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            model.save(
                str(models_dir / 'tier1_rf.pkl'),
                str(models_dir / 'tier2_iforest.pkl')
            )
            
            # Save preprocessor and selector
            if self.config['training'].get('save_preprocessor', True):
                joblib.dump(self.preprocessor, models_dir / 'preprocessor.pkl')
            if self.config['training'].get('save_feature_selector', True):
                joblib.dump(self.selector, models_dir / 'feature_selector.pkl')
        
        # Save config snapshot
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save metrics
        if self.config['output'].get('save_metrics', True):
            with open(self.output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save metadata
        metadata = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'dataset': self.config['dataset']['name'],
            'config_path': str(self.output_dir / 'config.yaml')
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save feature list alongside model artifacts
        if hasattr(self, 'selector') and self.selector is not None:
            feature_list = self.selector.get_selected_names()
            features_path = models_dir / 'features.json'
            with open(features_path, 'w') as f:
                json.dump({'features': feature_list, 'n_features': len(feature_list)}, f, indent=2)
            self.logger.info(f"Feature list saved: {features_path}")

        # Save CV scores if available
        if hasattr(self, 'cv_scores') and self.cv_scores is not None:
            metrics['cv_recall_mean'] = float(np.mean(self.cv_scores))
            metrics['cv_recall_std'] = float(np.std(self.cv_scores))
            metrics['cv_recall_scores'] = self.cv_scores

        self.logger.info(f"Artifacts saved to {self.output_dir}")

    def _log_to_mlflow(self, metrics: dict):
        """Log parameters, metrics, and artifacts to the active MLflow run."""
        # Log hyperparameters
        flat_params = {
            'dataset': self.config['dataset']['name'],
            'random_state': self.config['dataset'].get('random_state', 42),
            'smote': self.config['preprocessing'].get('apply_smote', True),
            'feature_selection_method': self.config['feature_selection'].get('method'),
            'n_features': self.config['feature_selection'].get('n_features'),
        }
        mlflow.log_params(flat_params)

        # Log scalar metrics
        loggable = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and v is not None
        }
        mlflow.log_metrics(loggable)

        # Log CV scores
        if hasattr(self, 'cv_scores') and self.cv_scores is not None:
            mlflow.log_metric('cv_recall_mean', float(np.mean(self.cv_scores)))
            mlflow.log_metric('cv_recall_std', float(np.std(self.cv_scores)))

        # Log artifacts directory
        mlflow.log_artifacts(str(self.output_dir))
        self.logger.info("MLflow logging complete")
