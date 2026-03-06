# Model Training and Evaluation Guide

This document describes the parameters, workflows, and strict configuration bounds required to continuously integrate datasets into the NIDS-ML algorithmic architecture.

## 1. Dataset Matrix Constraints

Data dimensions operating within NIDS-ML must map sequentially against labeled arrays. Supervised inputs require definitive attack categorization columns, while anomaly detection operations subset exclusively upon data tagged nominally as "Normal".

### Formats

- **NSL-KDD**: 41 continuous and discrete dimensions.
- **UNSW-NB15**: 49 dimensions detailing flow specifics.

Files strictly resolve against relative configurations mapped internally at `configs/datasets/`. Paths to `raw/` dependencies must conform to absolute or standardized localized directory references prior to loading.

## 2. Training Configuration Schemas

The operational behavior of the training pipeline is heavily parameterized within YAML structures logic located universally under `configs/training/`.

### Schema Map Example (`my_experiment.yaml`):

```yaml
experiment_name: production_cascade_v1

dataset:
  name: unsw_nb15
  config: configs/datasets/unsw_nb15.yaml
  test_size: 0.3
  random_state: 42
  stratify: true

preprocessing:
  apply_smote: false # Disabled strictly for ensembles utilizing internal sample balancing
  scaling_method: robust

feature_selection:
  method: combined # Uses Borda count fusion (RF + SHAP + MI)
  n_features: 25

models:
  use_stacking: true
  use_vae: true
  tier1:
    config: configs/models/stacking.yaml
  tier2:
    config: configs/models/vae.yaml

training:
  normal_label: Normal
  save_preprocessor: true
  save_feature_selector: true

evaluation:
  compute_shap: true
  calculate_mcc: true

output:
  base_dir: experiments/runs
  save_models: true
  save_metrics: true
  save_plots: true
```

## 3. Parameter Boundary Adjustments

### Tier 1: Stacking Ensemble (`configs/models/stacking.yaml`)

Tier 1 calculations prioritize non-linear boundaries. Hyperparameters require specific adjustments targeting dataset variability:

- **`n_estimators` (BalancedRandomForest)**: Controls baseline bootstrap magnitude. Scaled default is 200. Increases stabilize variance at latency costs.
- **`is_unbalance` (LGBM)**: Strictly boolean. Handles leaf-wise growth structures emphasizing specific minority classes inherent to U2R/R2L behaviors.
- **`C` (SVC)**: Regularization parameter smoothing geometric margin lines. Typically restricted between 0.1 and 1.0.

### Tier 2: VAE Fusion (`configs/models/vae.yaml`)

Anomaly detection requires precise dimensionality reduction.

- **`latent_dim`**: Establishes information bottlenecks. Extremely large datasets require bounds between 16 and 32 to prevent information overflow.
- **`threshold_percentile`**: Operates bounding cutoff limits for error distributions. Defaults uniformly to 95.0. Values > 99.0 significantly reduce FP thresholds and simultaneously reduce overall detection matrix scope.

## 4. Execution Logic

Deploy targeted parameter structures utilizing isolated environment runtimes:

```bash
python scripts/train.py --config configs/training/my_experiment.yaml
```

Metrics execution operates continuously over model finalization mapping explicit probability values:

```bash
python scripts/evaluate.py \
  --model experiments/runs/<iteration>/models \
  --dataset data/raw/validation_matrix.csv \
  --dataset-type unsw_nb15
```

## 5. Drift and Active Learning Configurations

Once a model establishes baseline constraints, it operates over drift-aware monitoring pipelines.

- **ADWIN Variances**: Instantiated during container deployment. Requires explicitly monitored variance arrays defining failure rates.
- **Feedback Sampling**: Generates uncertain variables against Shannon entropy, grouping outputs explicitly via K-Means vector distances. When `FeedbackBuffer` boundaries (default JSON config block limits) reach 100 samples, the CI/CD pipeline extracts verified labels and generates internal Github Actions triggers referencing localized re-training subroutines.
