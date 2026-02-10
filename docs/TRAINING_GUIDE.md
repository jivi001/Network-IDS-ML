# 🎓 Training Guide

This guide explains how to train the Hybrid NIDS model on your datasets.

---

## Table of Contents

1. [Prepare Your Dataset](#prepare-your-dataset)
2. [Configure Training](#configure-training)
3. [Run Training](#run-training)
4. [Monitor Progress](#monitor-progress)
5. [Evaluate Results](#evaluate-results)
6. [Advanced Training](#advanced-training)

---

## 1. Prepare Your Dataset

### Supported Datasets

The system supports three dataset formats:
- **NSL-KDD**: 41 features + label column
- **UNSW-NB15**: 49 features + label column
- **CIC-IDS2017**: 80 features + Label column

### Dataset Structure

Your dataset should be a CSV file with:
- **Features**: Network traffic characteristics (duration, bytes, flags, etc.)
- **Label column**: Attack type or "Normal"

Example:
```csv
duration,protocol_type,service,flag,src_bytes,dst_bytes,...,label
0,tcp,http,SF,181,5450,...,Normal
0,tcp,http,SF,239,486,...,DoS
```

### Place Dataset Files

```bash
# Create data directory structure
mkdir -p data/raw

# Copy your datasets
cp /path/to/nsl_kdd_train.csv data/raw/
cp /path/to/nsl_kdd_test.csv data/raw/
```

---

## 2. Configure Training

### Create Configuration File

Copy the default config and customize it:

```bash
cp configs/training/default.yaml configs/training/my_experiment.yaml
```

### Edit Configuration

Open `configs/training/my_experiment.yaml`:

```yaml
# Experiment name
experiment_name: nsl_kdd_baseline

# Dataset configuration
dataset:
  name: nsl_kdd
  config: configs/datasets/nsl_kdd.yaml
  test_size: 0.3
  random_state: 42
  stratify: true

# Preprocessing
preprocessing:
  apply_smote: true
  smote_strategy: auto
  smote_k_neighbors: 5
  scaling_method: standard

# Feature selection
feature_selection:
  method: rfe
  n_features: 20
  estimator: random_forest

# Model hyperparameters
models:
  tier1:
    type: random_forest
    config: configs/models/random_forest.yaml
  tier2:
    type: isolation_forest
    config: configs/models/isolation_forest.yaml

# Training settings
training:
  normal_label: Normal
  save_preprocessor: true
  save_feature_selector: true

# Evaluation
evaluation:
  compute_shap: true
  plot_confusion_matrix: true
  plot_pr_curve: true

# Output
output:
  base_dir: experiments/runs
  save_models: true
  save_metrics: true
  save_plots: true
```

### Update Dataset Config

Edit `configs/datasets/nsl_kdd.yaml`:

```yaml
dataset_type: nsl_kdd
description: NSL-KDD intrusion detection dataset

# Update paths to your actual files
paths:
  train: data/raw/nsl_kdd_train.csv
  test: data/raw/nsl_kdd_test.csv

# Label information
labels:
  normal: Normal
  attack_types:
    - DoS
    - Probe
    - R2L
    - U2R
```

---

## 3. Run Training

### Basic Training

```bash
python scripts/train.py --config configs/training/my_experiment.yaml
```

### With Custom Experiment Name

```bash
python scripts/train.py \
  --config configs/training/my_experiment.yaml \
  --experiment-name nsl_kdd_v2
```

### Expected Output

```
Starting experiment: nsl_kdd_baseline_20260210_214500

Loading data...
Train: (125973, 41), Test: (22544, 41)

Preprocessing...
[OK] Preprocessing complete: (125973, 41)

Applying SMOTE...
[SMOTE] 125973 -> 126000 samples
[OK] SMOTE complete

Feature selection...
[FeatureSelector] Starting RFE to select top 20 features...
[OK] Feature selection: (126000, 20)

Training Hybrid NIDS...
==================================================
TIER 1: Training Random Forest
==================================================
[Tier1-RF] Trained on 126000 samples, 20 features

==================================================
TIER 2: Training Isolation Forest
==================================================
[Tier2-iForest] Trained on 63000 normal samples
[Tier2-iForest] Anomaly threshold: -0.0234

[OK] Hybrid NIDS Training Complete

Evaluating...
======================================================================
  NIDS EVALUATION REPORT
======================================================================

--- Overall Metrics ---
Accuracy:  0.9523
Recall:    0.9523
Precision: 0.9180
F1-Score:  0.9348

Experiment complete: nsl_kdd_baseline_20260210_214500
Results saved to: experiments/runs/nsl_kdd_baseline_20260210_214500
```

---

## 4. Monitor Progress

### Check Experiment Directory

```bash
ls experiments/runs/nsl_kdd_baseline_20260210_214500/

# Output:
# config.yaml           # Config snapshot
# metadata.json         # Experiment metadata
# metrics.json          # Performance metrics
# models/               # Trained models
#   ├── tier1_rf.pkl
#   ├── tier2_iforest.pkl
#   ├── preprocessor.pkl
#   └── feature_selector.pkl
# plots/                # Visualizations
#   ├── confusion_matrix.png
#   ├── pr_curve.png
#   └── shap_summary.png
```

### View Metrics

```bash
cat experiments/runs/nsl_kdd_baseline_20260210_214500/metrics.json
```

```json
{
  "accuracy": 0.9523,
  "recall": 0.9523,
  "precision": 0.9180,
  "f1_score": 0.9348,
  "attack_detection_rate": 0.9523,
  "false_alarm_rate": 0.0820
}
```

---

## 5. Evaluate Results

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
  --model experiments/runs/nsl_kdd_baseline_20260210_214500/models \
  --dataset data/raw/nsl_kdd_test.csv \
  --dataset-type nsl_kdd
```

### Cross-Dataset Evaluation

Test generalization on a different dataset:

```bash
python scripts/cross_dataset_eval.py \
  --source-model experiments/runs/nsl_kdd_baseline_20260210_214500/models \
  --source-dataset nsl_kdd \
  --target-dataset unsw_nb15 \
  --target-data data/raw/unsw_nb15_test.csv \
  --output experiments/cross_dataset/nsl_to_unsw
```

---

## 6. Advanced Training

### Hyperparameter Tuning

Edit model configs to tune hyperparameters:

**Random Forest** (`configs/models/random_forest.yaml`):
```yaml
hyperparameters:
  n_estimators: 300        # Increase for better performance
  max_depth: 25            # Deeper trees
  min_samples_split: 3     # More granular splits
  class_weight: balanced
```

**Isolation Forest** (`configs/models/isolation_forest.yaml`):
```yaml
hyperparameters:
  contamination: 0.03      # Expected anomaly rate
  n_estimators: 300        # More trees for stability
```

### Custom Feature Selection

Modify feature selection settings:

```yaml
feature_selection:
  method: rfe
  n_features: 30           # Select more features
  estimator: random_forest
```

### Disable SMOTE

For balanced datasets:

```yaml
preprocessing:
  apply_smote: false
```

### Multiple Experiments

Run multiple experiments with different configs:

```bash
# Experiment 1: Baseline
python scripts/train.py --config configs/training/baseline.yaml

# Experiment 2: More features
python scripts/train.py --config configs/training/more_features.yaml

# Experiment 3: No SMOTE
python scripts/train.py --config configs/training/no_smote.yaml
```

---

## 📊 Training Tips

### 1. Start with Default Config
Use `configs/training/default.yaml` as a starting point.

### 2. Monitor Class Imbalance
Check class distribution in your dataset. Use SMOTE if imbalanced.

### 3. Feature Selection Impact
More features ≠ better performance. Start with 20, tune if needed.

### 4. Contamination Rate
Set Isolation Forest contamination to expected anomaly rate in production.

### 5. Save Everything
Always save preprocessor and feature selector for production deployment.

### 6. Experiment Tracking
Use descriptive experiment names to track different configurations.

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce dataset size or use chunked loading:
```python
# In custom training script
loader = DataLoader(dataset_type='nsl_kdd')
df = loader.load_csv('data/raw/nsl_kdd_train.csv', nrows=50000)
```

### Issue: Poor Performance

**Solutions**:
1. Increase `n_estimators` in model configs
2. Adjust feature selection (`n_features`)
3. Check class balance (use SMOTE if needed)
4. Verify dataset quality (no missing values)

### Issue: Training Too Slow

**Solutions**:
1. Reduce `n_estimators`
2. Set `n_jobs=-1` to use all CPU cores
3. Reduce dataset size for testing

---

## ✅ Next Steps

After training:
1. **Evaluate**: Test on holdout dataset
2. **Cross-validate**: Test on different datasets
3. **Deploy**: Move to production (see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md))
4. **Monitor**: Track performance in production

---

For production deployment, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
