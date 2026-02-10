# NIDS API Reference

This document provides a comprehensive reference for the Network Intrusion Detection System (NIDS) API.

## Table of Contents

- [Data Module](#data-module)
- [Preprocessing](#preprocessing)
- [Feature Selection](#feature-selection)
- [Models](#models)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Pipelines](#pipelines)
- [Utilities](#utilities)

---

## Data Module

### `nids.data.DataLoader`

Handles loading and basic validation of network intrusion datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017).

```python
from nids.data import DataLoader

loader = DataLoader(dataset_type='nsl-kdd')
df = loader.load_csv('path/to/data.csv')
X, y = loader.split_features_labels(df)
```

**Methods:**

- **`__init__(dataset_type: str = 'auto')`**
  - Initialize DataLoader. `dataset_type` can be 'nsl-kdd', 'unsw-nb15', 'cic-ids2017', or 'auto'.

- **`load_csv(filepath: str, chunksize: Optional[int] = None, nrows: Optional[int] = None) -> pd.DataFrame`**
  - Load CSV file with optional chunking.

- **`split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`**
  - Split DataFrame into features (X) and labels (y).

- **`get_dataset_info(df: pd.DataFrame) -> dict`**
  - Return summary statistics about the dataset.

### `nids.data.DatasetValidator`

Validates dataset schema and distributions to detect drift and data quality issues.

**Methods:**

- **`__init__(expected_schema: Optional[Dict[str, str]] = None)`**
  - Initialize validator with expected schema (column names to dtypes).

- **`validate(df: pd.DataFrame) -> List[str]`**
  - Validate dataset against expected schema. Returns list of error messages.

- **`detect_distribution_shift(reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.05) -> Dict`**
  - Detect distribution shift using Kolmogorov-Smirnov test.

---

## Preprocessing

### `nids.preprocessing.NIDSPreprocessor`

Unified preprocessing pipeline handling cleaning, label encoding, scaling, and SMOTE (training only).

```python
from nids.preprocessing import NIDSPreprocessor

preprocessor = NIDSPreprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
```

**Methods:**

- **`fit(X: pd.DataFrame, y: Optional[pd.Series] = None)`**
  - Fit all transformations (Imputer, LabelEncoder, StandardScaler) on training data.

- **`transform(X: pd.DataFrame) -> np.ndarray`**
  - Transform data using the fitted pipeline.

- **`apply_smote(X: np.ndarray, y: np.ndarray, sampling_strategy='auto', k_neighbors=5) -> Tuple[np.ndarray, np.ndarray]`**
  - Apply SMOTE to balance classes. **CRITICAL**: Use on training data ONLY.

- **`get_feature_names() -> List[str]`**
  - Return list of feature names in order.

---

## Feature Selection

### `nids.features.FeatureSelector`

Selects top features using Recursive Feature Elimination (RFE) with a Random Forest estimator.

**Methods:**

- **`__init__(n_features: int = 20, random_state: int = 42)`**
  - Initialize selector.

- **`fit(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None)`**
  - Fit RFE to select top `n_features`.

- **`transform(X: np.ndarray) -> np.ndarray`**
  - Reduce X to selected features.

- **`get_feature_importance_ranking() -> List[Tuple[str, float]]`**
  - Return sorted list of (feature_name, importance).

---

## Models

### `nids.models.HybridNIDS`

Two-tier cascade architecture:
1.  **Tier 1 (Random Forest)**: Detects known attacks.
2.  **Tier 2 (Isolation Forest)**: Detects zero-day anomalies in traffic classified as "Normal" by Tier 1.

```python
from nids.models import HybridNIDS

model = HybridNIDS()
model.train(X_train, y_train)
preds, flags = model.predict(X_test)
```

**Methods:**

- **`train(X_train: np.ndarray, y_train: np.ndarray, normal_label: str = 'Normal')`**
  - Train both tiers. Tier 2 is trained only on Normal samples.

- **`predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**
  - Cascade prediction. Returns `(final_labels, tier_flags)`. 
  - `tier_flags`: 1 (Tensor 1 decision), 2 (Tier 2 decision).

- **`predict_with_scores(X: np.ndarray) -> Dict`**
  - Returns detailed predictions including probabilities and anomaly scores.

- **`save(tier1_path: str, tier2_path: str)`**
  - Save both models to disk.

- **`load(tier1_path: str, tier2_path: str, normal_label: str)`**
  - Load models from disk.

### `nids.models.SupervisedModel`

Wrapper for Tier 1 Random Forest classifier.

### `nids.models.UnsupervisedModel`

Wrapper for Tier 2 Isolation Forest anomaly detector.

---

## Evaluation

### `nids.evaluation.NIDSEvaluator`

Security-focused evaluation suite prioritizing Recall (detection rate) and F2-Score.

**Methods:**

- **`evaluate(y_true, y_pred, y_proba=None, labels=None, normal_label='Normal') -> Dict`**
  - Comprehensive evaluation returning accuracy, precision, recall, f1, f2, and confusion matrix.

- **`plot_feature_importance(importances, feature_names, top_n=20)`**
  - Generate feature importance plot.

### `nids.evaluation.StatisticalEvaluator`

Implements statistical significance testing (Paired T-test, Wilcoxon) and repeated k-fold cross-validation.

### `nids.evaluation.CrossDatasetEvaluator`

Evaluates model generalization by training on one dataset and testing on another (e.g., Train on NSL-KDD, Test on CIC-IDS2017).

---

## Explainability

### `nids.explainability.SHAPExplainer`

SHAP-based explainability for model interpretability.

**Methods:**

- **`explain_prediction(model, X_sample, feature_names) -> Dict`**
  - Generate SHAP values for a single prediction.

- **`plot_feature_importance(model, X, feature_names, output_path=None)`**
  - Plot global SHAP feature importance.

---

## Pipelines

### `nids.pipelines.TrainingPipeline`

End-to-end training workflow: Data Loading -> Preprocessing -> SMOTE -> Feature Selection -> Model Training -> Evaluation.

```python
from nids.pipelines import TrainingPipeline

pipeline = TrainingPipeline('configs/config.yaml')
pipeline.run()
```

### `nids.pipelines.InferencePipeline`

Production inference pipeline for simplified prediction on new data.

```python
from nids.pipelines import InferencePipeline

pipeline = InferencePipeline(model_version='v1.0.0')
result = pipeline.predict_single(feature_vector)
```

### `nids.pipelines.EvaluationPipeline`

Pipeline for assessing trained models on independent test sets.

---

## Utilities

### `nids.utils.config`

- **`load_config(path) -> Dict`**: Load YAML config.
- **`save_config(config, path)`**: Save dict to YAML.

### `nids.utils.logging`

- **`setup_logger(name, log_file, level)`**: Configure logging with console and file handlers.
