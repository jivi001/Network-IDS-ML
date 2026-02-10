# Repository Structure Refactoring Proposal
## Network-IDS-ML: Production-Grade ML Project Organization

---

## 1. Proposed Directory Structure

```
Network-IDS-ML/
│
├── .github/                          # GitHub-specific configurations
│   └── workflows/
│       ├── ci.yml                    # Continuous integration (linting, tests)
│       └── model-training.yml        # Automated model training pipeline
│
├── configs/                          # Configuration files (YAML/JSON)
│   ├── datasets/
│   │   ├── nsl_kdd.yaml             # NSL-KDD dataset schema and paths
│   │   ├── unsw_nb15.yaml           # UNSW-NB15 configuration
│   │   └── cic_ids2017.yaml         # CIC-IDS2017 configuration
│   ├── models/
│   │   ├── random_forest.yaml       # RF hyperparameters
│   │   ├── isolation_forest.yaml    # iForest hyperparameters
│   │   └── svm.yaml                 # SVM baseline config
│   ├── preprocessing.yaml            # Preprocessing pipeline settings
│   └── training.yaml                 # Training loop configuration
│
├── data/                             # Data storage (gitignored except README)
│   ├── raw/                          # Original datasets (downloaded)
│   │   ├── NSL-KDD/
│   │   ├── UNSW-NB15/
│   │   └── CIC-IDS2017/
│   ├── processed/                    # Preprocessed datasets (cleaned, encoded)
│   │   ├── nsl_kdd_train.pkl
│   │   ├── nsl_kdd_test.pkl
│   │   └── ...
│   ├── interim/                      # Intermediate data (feature-selected, SMOTE)
│   │   └── nsl_kdd_train_balanced.pkl
│   └── README.md                     # Data acquisition instructions
│
├── deployment/                       # Production deployment artifacts
│   ├── docker/
│   │   ├── Dockerfile                # Container for NIDS service
│   │   └── docker-compose.yml        # Multi-service orchestration
│   ├── kubernetes/
│   │   ├── deployment.yaml           # K8s deployment spec
│   │   └── service.yaml              # K8s service exposure
│   └── scripts/
│       ├── deploy.sh                 # Deployment automation script
│       └── health_check.py           # Service health monitoring
│
├── docs/                             # Documentation
│   ├── methodology.md                # Research methodology (TASK 1 output)
│   ├── architecture.md               # System architecture diagrams
│   ├── api_reference.md              # API documentation
│   ├── dataset_guide.md              # Dataset download and preparation
│   └── deployment_guide.md           # Production deployment instructions
│
├── experiments/                      # Experiment tracking and results
│   ├── runs/                         # Individual experiment runs
│   │   ├── exp_001_baseline_rf/
│   │   │   ├── config.yaml           # Experiment configuration snapshot
│   │   │   ├── metrics.json          # Evaluation metrics
│   │   │   ├── confusion_matrix.png  # Visualizations
│   │   │   └── model.pkl             # Trained model checkpoint
│   │   ├── exp_002_smote_tuning/
│   │   └── exp_003_hybrid_cascade/
│   ├── notebooks/                    # Jupyter notebooks for analysis
│   │   ├── 01_eda_nsl_kdd.ipynb     # Exploratory data analysis
│   │   ├── 02_feature_importance.ipynb
│   │   ├── 03_hyperparameter_tuning.ipynb
│   │   └── 04_cross_dataset_eval.ipynb
│   └── results/                      # Aggregated results (tables, plots)
│       ├── comparison_table.csv      # Model comparison metrics
│       └── pr_curves.png             # Precision-Recall curves
│
├── models/                           # Saved production models
│   ├── production/
│   │   ├── tier1_rf.pkl              # Production Random Forest
│   │   ├── tier2_iforest.pkl         # Production Isolation Forest
│   │   └── preprocessor.pkl          # Fitted preprocessing pipeline
│   └── baselines/
│       ├── svm_baseline.pkl
│       └── logistic_regression.pkl
│
├── nids/                             # Core Python package (main library)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                # Dataset loading (CSV, Parquet, SQL)
│   │   ├── schemas.py                # Dataset schema definitions
│   │   └── downloaders.py            # Automated dataset download scripts
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaners.py               # Data cleaning (inf, NaN handling)
│   │   ├── encoders.py               # Categorical encoding
│   │   ├── scalers.py                # Feature scaling (StandardScaler)
│   │   └── balancers.py              # SMOTE and class balancing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── selection.py              # Feature selection (RF importance, RFE)
│   │   └── engineering.py            # Custom feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base model interface (abstract class)
│   │   ├── supervised.py             # Random Forest, SVM wrappers
│   │   ├── unsupervised.py           # Isolation Forest wrapper
│   │   └── hybrid.py                 # Hybrid cascade system
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Precision, Recall, F1, PR-AUC
│   │   └── visualizations.py         # Confusion matrix, PR curves
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration loading (YAML/JSON)
│   │   ├── logging.py                # Logging setup
│   │   └── io.py                     # File I/O utilities (pickle, joblib)
│   └── pipelines/
│       ├── __init__.py
│       ├── training.py               # End-to-end training pipeline
│       ├── inference.py              # Inference pipeline
│       └── evaluation.py             # Evaluation pipeline
│
├── scripts/                          # Standalone executable scripts
│   ├── download_datasets.py          # Download NSL-KDD, UNSW-NB15, CIC-IDS2017
│   ├── preprocess_data.py            # Run preprocessing pipeline
│   ├── train_model.py                # Train models from config
│   ├── evaluate_model.py             # Evaluate saved models
│   ├── run_experiment.py             # Execute experiment with tracking
│   └── export_model.py               # Export model for deployment (ONNX, TFLite)
│
├── tests/                            # Unit and integration tests
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_feature_selection.py
│   │   ├── test_models.py
│   │   └── test_hybrid_system.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_inference_pipeline.py
│   └── fixtures/
│       └── sample_data.csv           # Small test dataset
│
├── .gitignore                        # Git ignore rules
├── .dockerignore                     # Docker ignore rules
├── LICENSE                           # MIT License
├── README.md                         # Project overview and quick start
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies (pytest, black, flake8)
├── setup.py                          # Package installation script
└── pyproject.toml                    # Modern Python project metadata (PEP 518)
```

---

## 2. Folder and File Purposes

### 2.1 Core Library (`nids/`)

**Purpose**: Reusable, modular Python package containing all ML logic.

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `data/` | Dataset loading and management | `loaders.py`: CSV/Parquet readers with schema validation<br>`schemas.py`: Column name mappings for NSL-KDD, UNSW-NB15, CIC-IDS2017<br>`downloaders.py`: Automated download from public repositories |
| `preprocessing/` | Data cleaning and transformation | `cleaners.py`: `handle_inf()`, `impute_missing()`<br>`encoders.py`: `LabelEncoder`, `OneHotEncoder` wrappers<br>`scalers.py`: `StandardScaler`, `MinMaxScaler`<br>`balancers.py`: `SMOTEBalancer` class |
| `features/` | Feature engineering and selection | `selection.py`: `RFFeatureSelector`, `RFESelector`<br>`engineering.py`: Custom feature derivation (e.g., flow ratios) |
| `models/` | ML model implementations | `base.py`: `BaseModel` abstract class (train, predict, save, load)<br>`supervised.py`: `RandomForestModel`, `SVMModel`<br>`unsupervised.py`: `IsolationForestModel`<br>`hybrid.py`: `HybridNIDS` cascade system |
| `evaluation/` | Performance assessment | `metrics.py`: `calculate_metrics()`, `confusion_matrix()`<br>`visualizations.py`: `plot_confusion_matrix()`, `plot_pr_curve()` |
| `utils/` | Shared utilities | `config.py`: YAML/JSON config parser<br>`logging.py`: Structured logging setup<br>`io.py`: Model serialization (pickle, joblib) |
| `pipelines/` | End-to-end workflows | `training.py`: `TrainingPipeline` class<br>`inference.py`: `InferencePipeline` class<br>`evaluation.py`: `EvaluationPipeline` class |

### 2.2 Configuration (`configs/`)

**Purpose**: Centralized, version-controlled hyperparameters and settings.

- **Dataset configs**: Column mappings, file paths, train/test splits
- **Model configs**: Hyperparameters (n_estimators, max_depth, contamination)
- **Preprocessing configs**: SMOTE parameters, scaling methods
- **Training configs**: Batch size, random seed, cross-validation folds

**Format**: YAML (human-readable, supports comments)

**Example** (`configs/models/random_forest.yaml`):
```yaml
model_type: RandomForest
hyperparameters:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  criterion: gini
  class_weight: balanced
  random_state: 42
```

### 2.3 Data (`data/`)

**Purpose**: Organized data storage with clear lineage.

- `raw/`: Immutable original datasets (never modified)
- `processed/`: Cleaned, encoded, scaled data (ready for training)
- `interim/`: Intermediate artifacts (SMOTE-balanced, feature-selected)

**Gitignore**: All data files ignored except `README.md` (instructions for obtaining datasets)

### 2.4 Experiments (`experiments/`)

**Purpose**: Reproducible experiment tracking.

- `runs/`: Each experiment gets unique directory with:
  - Configuration snapshot (ensures reproducibility)
  - Metrics (JSON for programmatic access)
  - Visualizations (PNG/PDF for reports)
  - Model checkpoint
- `notebooks/`: Interactive analysis (EDA, hyperparameter tuning)
- `results/`: Aggregated comparisons across experiments

**Naming Convention**: `exp_<number>_<description>` (e.g., `exp_042_smote_k5_vs_k10`)

### 2.5 Scripts (`scripts/`)

**Purpose**: Command-line entry points for common tasks.

**Example Usage**:
```bash
# Download datasets
python scripts/download_datasets.py --datasets nsl-kdd unsw-nb15

# Preprocess data
python scripts/preprocess_data.py --config configs/preprocessing.yaml --input data/raw/NSL-KDD/train.csv --output data/processed/nsl_kdd_train.pkl

# Train model
python scripts/train_model.py --config configs/models/random_forest.yaml --data data/interim/nsl_kdd_train_balanced.pkl --output models/production/tier1_rf.pkl

# Evaluate model
python scripts/evaluate_model.py --model models/production/tier1_rf.pkl --test-data data/processed/nsl_kdd_test.pkl --output experiments/results/rf_eval.json
```

### 2.6 Deployment (`deployment/`)

**Purpose**: Production deployment artifacts.

- **Docker**: Containerized NIDS service for cloud/on-prem deployment
- **Kubernetes**: Scalable orchestration for high-throughput networks
- **Scripts**: Automated deployment, health checks, monitoring integration

### 2.7 Tests (`tests/`)

**Purpose**: Ensure code correctness and prevent regressions.

- `unit/`: Test individual functions (preprocessing, feature selection)
- `integration/`: Test full pipelines (training, inference)
- `fixtures/`: Small sample datasets for fast testing

**Run Tests**:
```bash
pytest tests/ --cov=nids --cov-report=html
```

---

## 3. Python Module Naming Conventions

### 3.1 Import Structure

```python
# Core library imports
from nids.data.loaders import DataLoader
from nids.preprocessing.cleaners import handle_infinite_values
from nids.preprocessing.balancers import SMOTEBalancer
from nids.features.selection import RFFeatureSelector
from nids.models.supervised import RandomForestModel
from nids.models.unsupervised import IsolationForestModel
from nids.models.hybrid import HybridNIDS
from nids.evaluation.metrics import calculate_classification_metrics
from nids.evaluation.visualizations import plot_confusion_matrix
from nids.pipelines.training import TrainingPipeline
from nids.utils.config import load_config
```

### 3.2 Class Naming

| Component | Class Name | File Location |
|-----------|-----------|---------------|
| Data Loader | `DataLoader` | `nids/data/loaders.py` |
| Preprocessor | `NIDSPreprocessor` | `nids/preprocessing/cleaners.py` |
| SMOTE Balancer | `SMOTEBalancer` | `nids/preprocessing/balancers.py` |
| Feature Selector | `RFFeatureSelector` | `nids/features/selection.py` |
| Random Forest | `RandomForestModel` | `nids/models/supervised.py` |
| SVM | `SVMModel` | `nids/models/supervised.py` |
| Isolation Forest | `IsolationForestModel` | `nids/models/unsupervised.py` |
| Hybrid System | `HybridNIDS` | `nids/models/hybrid.py` |
| Evaluator | `ModelEvaluator` | `nids/evaluation/metrics.py` |
| Training Pipeline | `TrainingPipeline` | `nids/pipelines/training.py` |
| Inference Pipeline | `InferencePipeline` | `nids/pipelines/inference.py` |

---

## 4. Component Placement Guide

### 4.1 Preprocessing Steps

| Step | Current Location | New Location | Module Name |
|------|-----------------|--------------|-------------|
| Infinite value handling | `src/preprocessing.py` | `nids/preprocessing/cleaners.py` | `handle_infinite_values()` |
| NaN imputation | `src/preprocessing.py` | `nids/preprocessing/cleaners.py` | `impute_missing_values()` |
| Label encoding | `src/preprocessing.py` | `nids/preprocessing/encoders.py` | `LabelEncoderWrapper` |
| Scaling | `src/preprocessing.py` | `nids/preprocessing/scalers.py` | `StandardScalerWrapper` |
| SMOTE | `src/preprocessing.py` | `nids/preprocessing/balancers.py` | `SMOTEBalancer` |

### 4.2 Model Implementations

| Model | Current Location | New Location | Class Name |
|-------|-----------------|--------------|------------|
| Random Forest | `src/models.py` | `nids/models/supervised.py` | `RandomForestModel` |
| SVM | `src/models.py` | `nids/models/supervised.py` | `SVMModel` |
| Isolation Forest | `src/models.py` | `nids/models/unsupervised.py` | `IsolationForestModel` |
| Hybrid System | `src/hybrid_system.py` | `nids/models/hybrid.py` | `HybridNIDS` |

### 4.3 Training and Evaluation

| Component | Current Location | New Location | Purpose |
|-----------|-----------------|--------------|---------|
| Training loop | `main.py` | `nids/pipelines/training.py` | Reusable training pipeline |
| Evaluation metrics | `src/evaluation.py` | `nids/evaluation/metrics.py` | Metric calculation |
| Visualizations | `src/evaluation.py` | `nids/evaluation/visualizations.py` | Plotting functions |
| Experiment runner | `main.py` | `scripts/run_experiment.py` | CLI experiment execution |

### 4.4 Utilities

| Utility | Current Location | New Location | Purpose |
|---------|-----------------|--------------|---------|
| Data loading | `src/data_loader.py` | `nids/data/loaders.py` | CSV/Parquet loading |
| Dataset download | `src/download_datasets.py` | `nids/data/downloaders.py` | Automated downloads |
| Feature selection | `src/feature_selection.py` | `nids/features/selection.py` | RF importance, RFE |

---

## 5. Training, Evaluation, and Experiments

### 5.1 Training Workflow

**Location**: `nids/pipelines/training.py`

**Class**: `TrainingPipeline`

**Responsibilities**:
1. Load configuration from `configs/`
2. Load and preprocess data
3. Apply SMOTE balancing
4. Perform feature selection
5. Train models (Tier 1 and Tier 2)
6. Save models to `models/production/`
7. Log metrics to `experiments/runs/<exp_id>/`

**Usage**:
```python
from nids.pipelines.training import TrainingPipeline

pipeline = TrainingPipeline(config_path='configs/training.yaml')
pipeline.run(
    train_data_path='data/processed/nsl_kdd_train.pkl',
    output_dir='experiments/runs/exp_050_final_model'
)
```

### 5.2 Evaluation Workflow

**Location**: `nids/pipelines/evaluation.py`

**Class**: `EvaluationPipeline`

**Responsibilities**:
1. Load trained models
2. Load test data
3. Generate predictions
4. Calculate metrics (Precision, Recall, F1, PR-AUC)
5. Generate visualizations (confusion matrix, PR curves)
6. Save results to `experiments/results/`

**Usage**:
```python
from nids.pipelines.evaluation import EvaluationPipeline

pipeline = EvaluationPipeline(
    model_path='models/production/tier1_rf.pkl',
    test_data_path='data/processed/nsl_kdd_test.pkl'
)
results = pipeline.run(output_dir='experiments/results/')
```

### 5.3 Experiment Tracking

**Location**: `experiments/runs/<exp_id>/`

**Structure**:
```
experiments/runs/exp_042_smote_k5/
├── config.yaml                # Snapshot of all configs used
├── metrics.json               # {precision: 0.92, recall: 0.95, ...}
├── confusion_matrix.png
├── pr_curve.png
├── feature_importance.png
├── tier1_rf.pkl               # Model checkpoint
├── tier2_iforest.pkl
└── logs.txt                   # Training logs
```

**Automation**:
```bash
python scripts/run_experiment.py \
    --config configs/training.yaml \
    --dataset nsl-kdd \
    --exp-name "smote_k5_vs_k10" \
    --output experiments/runs/exp_042_smote_k5
```

---

## 6. Migration Guide from Current Structure

### 6.1 File Mapping

| Current File | New Location | Notes |
|--------------|-------------|-------|
| `src/data_loader.py` | `nids/data/loaders.py` | Rename class to `DataLoader` |
| `src/preprocessing.py` | `nids/preprocessing/` | Split into `cleaners.py`, `encoders.py`, `scalers.py`, `balancers.py` |
| `src/feature_selection.py` | `nids/features/selection.py` | No changes |
| `src/models.py` | `nids/models/supervised.py` + `nids/models/unsupervised.py` | Split by learning type |
| `src/hybrid_system.py` | `nids/models/hybrid.py` | No changes |
| `src/evaluation.py` | `nids/evaluation/metrics.py` + `nids/evaluation/visualizations.py` | Split metrics and plotting |
| `src/download_datasets.py` | `nids/data/downloaders.py` | No changes |
| `main.py` | `scripts/run_experiment.py` | Refactor into reusable pipeline |

### 6.2 Migration Steps

1. **Create new directory structure**:
   ```bash
   mkdir -p nids/{data,preprocessing,features,models,evaluation,utils,pipelines}
   mkdir -p configs/{datasets,models}
   mkdir -p experiments/{runs,notebooks,results}
   mkdir -p scripts tests deployment/docker docs
   ```

2. **Move and refactor files**:
   - Split monolithic files (`preprocessing.py`, `models.py`, `evaluation.py`)
   - Add `__init__.py` to all package directories
   - Update import statements

3. **Create configuration files**:
   - Extract hardcoded hyperparameters to YAML configs
   - Create dataset schema definitions

4. **Update `setup.py`**:
   ```python
   from setuptools import setup, find_packages
   
   setup(
       name='nids',
       version='1.0.0',
       packages=find_packages(),
       install_requires=[
           'numpy>=1.21.0',
           'pandas>=1.3.0',
           'scikit-learn>=1.0.0',
           'imbalanced-learn>=0.9.0',
           'matplotlib>=3.4.0',
           'seaborn>=0.11.0',
           'pyyaml>=5.4.0'
       ]
   )
   ```

5. **Install package in editable mode**:
   ```bash
   pip install -e .
   ```

6. **Update tests**:
   - Refactor tests to match new module structure
   - Run `pytest tests/` to verify

---

## 7. Benefits of Refactored Structure

### 7.1 Modularity
- **Separation of concerns**: Each module has single responsibility
- **Reusability**: Import only needed components
- **Testability**: Isolated units easier to test

### 7.2 Reproducibility
- **Configuration management**: All hyperparameters version-controlled
- **Experiment tracking**: Each run fully documented
- **Data lineage**: Clear raw → processed → interim flow

### 7.3 Research-Ready
- **Notebook integration**: Jupyter notebooks in `experiments/notebooks/`
- **Comparison framework**: Standardized metrics across experiments
- **Publication support**: `docs/methodology.md` ready for paper submission

### 7.4 SOC/Industry Alignment
- **Deployment artifacts**: Docker, Kubernetes configs included
- **Production models**: Separate directory for deployed models
- **Monitoring**: Health checks and logging infrastructure
- **Scalability**: Pipeline design supports distributed training

### 7.5 Collaboration
- **Clear onboarding**: `README.md` + `docs/` guide new contributors
- **Code quality**: `tests/` + CI/CD ensure stability
- **Standardization**: Consistent naming and structure

---

## 8. Recommended Next Steps

1. **Phase 1: Core Refactoring** (1-2 days)
   - Create directory structure
   - Move files to new locations
   - Update imports

2. **Phase 2: Configuration Externalization** (1 day)
   - Create YAML configs
   - Refactor hardcoded parameters
   - Implement config loading utility

3. **Phase 3: Pipeline Development** (2-3 days)
   - Implement `TrainingPipeline`
   - Implement `EvaluationPipeline`
   - Create experiment runner script

4. **Phase 4: Testing and Documentation** (1-2 days)
   - Write unit tests
   - Update README.md
   - Document API in `docs/api_reference.md`

5. **Phase 5: Deployment Preparation** (2-3 days)
   - Create Dockerfile
   - Write deployment scripts
   - Set up CI/CD pipeline

---

**Total Estimated Effort**: 7-11 days for full refactoring

**Priority**: Start with Phase 1 (core refactoring) to establish foundation, then iterate on remaining phases based on project timeline.
