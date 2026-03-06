# NIDS-ML: Complete Technical Reference

> **Network Intrusion Detection System — Machine Learning Platform**
> Version 2.0 | Architecture: 4-Tier Hybrid Cascade

---

## 1. System Overview

The NIDS-ML platform detects and classifies malicious network traffic using a multi-layer machine learning architecture. It is designed to be:

| Property             | How Achieved                                               |
| -------------------- | ---------------------------------------------------------- |
| **Accurate**         | Stacking ensemble (3 base learners + meta-learner)         |
| **Zero-day capable** | VAE + IsolationForest fusion anomaly detection             |
| **Explainable**      | Per-prediction SHAP values exposed via API                 |
| **Adaptive**         | ADWIN concept drift detector triggers automated retraining |
| **SOC-ready**        | FastAPI server, WebSocket live feed, analyst feedback loop |
| **Reproducible**     | MLflow experiment tracking + DVC dataset versioning        |

---

## 2. ML Architecture — 4-Tier Hybrid Cascade

```
                     NETWORK TRAFFIC
                           │
                           ▼
              ┌────────────────────────┐
              │  TIER 1: Stacking      │   ← Known attack classifier
              │  Ensemble              │
              │  BRF + LightGBM + SVC  │
              │  → LogReg meta-learner │
              └────────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          │ High-conf Attack ≥0.90  │ Low-conf / Normal
          ▼                         ▼
     🚨 ALERT            ┌──────────────────────┐
                         │  TIER 2: Fusion      │   ← Zero-day detector
                         │  Anomaly Detector    │
                         │  VAE + IsolationForest│
                         └────────┬─────────────┘
                                  │
                    ┌─────────────┴────────────┐
                    │ Anomaly score > threshold │ Score ≤ threshold
                    ▼                           ▼
               ⚠️ ZERO-DAY               ✅ NORMAL
                    │
                    ▼
         ┌─────────────────────┐
         │  TIER 3:            │   ← Per-alert explainability
         │  SHAP Explainer     │
         │  + LIME fallback    │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  TIER 4:            │   ← Closed-loop adaptation
         │  ADWIN Drift Detect │
         │  + Active Learning  │
         │  + SOC Feedback     │
         └─────────────────────┘
```

---

## 3. Algorithms & Parameters

### 3.1 Tier 1 — Stacking Ensemble

**File:** `nids/models/stacking.py`

| Component    | Algorithm                        | Key Parameters                                                                    |
| ------------ | -------------------------------- | --------------------------------------------------------------------------------- |
| Base 1       | `BalancedRandomForestClassifier` | `n_estimators=200`, `max_depth=20`, `criterion='gini'`, `sampling_strategy='all'` |
| Base 2       | `LGBMClassifier` (LightGBM)      | `n_estimators=300`, `num_leaves=63`, `learning_rate=0.05`, `is_unbalance=True`    |
| Base 3       | `CalibratedClassifierCV(SVC)`    | `kernel='rbf'`, `C=10.0`, `gamma='scale'`, `cv=3`                                 |
| Meta-learner | `LogisticRegression`             | `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `passthrough=True`                    |

**How it works:**

- All 3 base learners are fitted with 5-fold cross-fitting (`cv=5`) so the meta-learner trains on out-of-fold predictions
- `passthrough=True` means the meta-learner also sees the original features — not just base predictions
- Class imbalance is handled natively: BRF via bootstrap, LightGBM via `is_unbalance`, SVC via calibration
- Produces calibrated probability estimates for all attack classes

**Default mode (backward-compatible):** `BalancedRandomForestClassifier` only (`use_stacking=False`)

---

### 3.2 Tier 2 — Fusion Anomaly Detector

**File:** `nids/models/anomaly.py`

#### VAE (Variational Autoencoder)

| Layer        | Architecture                                                          |
| ------------ | --------------------------------------------------------------------- |
| Encoder      | `Linear(n_features→64) → LayerNorm → ReLU → Linear(64→32) → ReLU`     |
| Latent space | `Linear(32→16)` for μ, `Linear(32→16)` for log-σ²                     |
| Decoder      | `Linear(16→32) → ReLU → Linear(32→64) → ReLU → Linear(64→n_features)` |

| Parameter              | Value | Effect                                  |
| ---------------------- | ----- | --------------------------------------- |
| `latent_dim`           | 16    | Compressed representation size          |
| `epochs`               | 50    | Training iterations                     |
| `batch_size`           | 256   | Mini-batch size                         |
| `learning_rate`        | 1e-3  | Adam optimizer LR                       |
| `beta`                 | 1.0   | KL divergence weight (β-VAE)            |
| `threshold_percentile` | 95.0  | Anomaly threshold = p95 of train errors |

**Loss function:** `ELBO = MSE(x, x̂) + β × KL(q(z|x) ‖ p(z))`

**Zero-day score:** `reconstruction_error(x) = mean((x − x̂)²)` — higher = more anomalous

#### IsolationForest

| Parameter       | Value                           |
| --------------- | ------------------------------- |
| `n_estimators`  | 200                             |
| `contamination` | 0.05 (5% expected anomaly rate) |

#### Fusion

```
fused_score = 0.5 × norm(VAE_error) + 0.5 × norm(IForest_score)
anomaly if fused_score > p95(training fused scores)
```

Both sub-scores are min-max normalised to [0,1] before weighting, so neither detector dominates.

---

### 3.3 Tier 3 — Explainability Engine

**File:** `nids/explainability/`

| Method   | Algorithm            | Speed       | Use Case                        |
| -------- | -------------------- | ----------- | ------------------------------- |
| Primary  | `shap.TreeExplainer` | Fast (~2ms) | All tree models (BRF, LightGBM) |
| Fallback | `LIME`               | Moderate    | Black-box or SVC predictions    |

Output: `shap_values[i]` = contribution of each feature to the prediction for sample `i`

---

### 3.4 Tier 4 — Adaptive Retraining

**Files:** `nids/drift/detector.py`, `nids/active_learning/`

#### ADWIN Drift Detector

| Parameter     | Value | Effect                               |
| ------------- | ----- | ------------------------------------ |
| `delta`       | 0.002 | Confidence; smaller = more sensitive |
| `min_samples` | 30    | Samples before drift can be declared |

**How it works:** Maintains a sliding window of per-sample error (0/1). Declares drift when the mean error in the left sub-window significantly differs from the right sub-window (CUSUM test with adaptive window).

**Fallback:** Page-Hinkley test (if `river` not installed)

#### Active Learning Query Strategy

| Parameter          | Default | Effect                                  |
| ------------------ | ------- | --------------------------------------- |
| `budget`           | 50      | Max samples sent to analyst per round   |
| `uncertainty_pool` | 500     | Pre-filter by entropy before clustering |

**Algorithm:**

1. **Uncertainty:** compute Shannon entropy of Tier 1 probabilities for each unlabeled sample
2. **Diversity:** run `MiniBatchKMeans(k=budget)` on the top-500 most uncertain samples
3. **Selection:** from each cluster, pick the sample with highest entropy
4. Present those 50 samples to the SOC analyst for labeling

#### Feedback Buffer

| Parameter      | Default                |
| -------------- | ---------------------- |
| `trigger_size` | 500                    |
| `buffer_path`  | `feedback_buffer.json` |

SOC actions → `approve` (TP confirmed), `reject` (FP → relabeled Normal), `relabel` (wrong class)

---

## 4. Data Pipeline

### 4.1 Datasets

| Dataset         | Traffic Type            | Size       | Features    |
| --------------- | ----------------------- | ---------- | ----------- |
| **NSL-KDD**     | Synthetic network logs  | ~150K rows | 41 features |
| **UNSW-NB15**   | Modern network flows    | ~2.5M rows | 49 features |
| **CIC-IDS2017** | Real-world PCAP-derived | ~2.8M rows | 78 features |

### 4.2 Training Pipeline — Step by Step

```
configs/training/hardened_rf.yaml
         │
         ▼
Step 1: DataLoader
        └─ Load CSV (chunked for large files)
        └─ Split features / label column
        └─ Train/test split (stratified, test_size=0.3)

Step 2: NIDSPreprocessor
        └─ Impute NaN → median strategy
        └─ Encode categoricals → OneHotEncoder
        └─ Scale numerics → RobustScaler (outlier-robust)
        └─ fit_transform(X_train), transform(X_test) (no leakage)

Step 3: FeatureSelector (method configurable)
        ├─ 'importance'  → RF Gini importance, top-N
        ├─ 'rfe'         → Recursive Feature Elimination
        ├─ 'shap'        → SHAP TreeExplainer mean |SHAP|
        ├─ 'mutual_info' → Shannon MI between features and label
        └─ 'combined'    → Borda count fusion of all 3 rankers

Step 4: HybridNIDS.train()
        ├─ Tier 1: StackingEnsemble.train(X_all, y_all)
        │          5-fold cross-fitting for base learners
        └─ Tier 2: FusionAnomalyDetector.train(X_normal_only)
                   VAE gradient descent + IForest fitting

Step 5: NIDSEvaluator.evaluate()
        └─ See Section 5
```

### 4.3 Feature Selection Parameters

| Method          | n_features | Recommended For                  |
| --------------- | ---------- | -------------------------------- |
| `'combined'`    | 20–30      | Production (best reliability)    |
| `'shap'`        | 20–30      | IEEE publication-grade analysis  |
| `'mutual_info'` | 20–30      | Fast, no model dependency        |
| `'importance'`  | 15–25      | Baseline, very fast              |
| `'rfe'`         | 15–20      | When exact feature count matters |

### 4.4 Class Imbalance Handling

`BalancedRandomForestClassifier` resamples each bootstrap sample so all classes appear equally — no explicit SMOTE step needed. For the stacking base learners:

- BRF: `sampling_strategy='all'` + `replacement=True`
- LightGBM: `is_unbalance=True`
- SVC: `CalibratedClassifierCV` with probability calibration

---

## 5. Evaluation Framework

**File:** `nids/evaluation/metrics.py` → `NIDSEvaluator`

### 5.1 Full Metric Set

| Metric                    | Formula                           | Target  | Priority                               |
| ------------------------- | --------------------------------- | ------- | -------------------------------------- |
| **PR-AUC**                | Area under Precision-Recall curve | > 0.97  | Critical (imbalanced datasets)         |
| **F2-Score**              | `(5·P·R) / (4P+R)`                | > 0.95  | Primary production KPI (recall-biased) |
| **MCC**                   | `(TP·TN − FP·FN) / √(...)`        | > 0.92  | Best single discriminator              |
| **ROC-AUC**               | Area under ROC curve              | > 0.99  | Benchmark comparison                   |
| **Attack Detection Rate** | `TP / (TP + FN)`                  | > 0.99  | Miss rate (critical)                   |
| **False Alarm Rate**      | `FP / (FP + TN)`                  | < 0.02  | SOC workload                           |
| **Alert Fatigue Index**   | `FP / (TP + FP + ε)`              | < 0.05  | Analyst burnout risk                   |
| **Detect Latency**        | ms/sample                         | < 10 ms | Real-time constraint                   |

### 5.2 Per-Attack-Family Metrics

Pass `attack_families` dict to `evaluate()`:

```python
attack_families = {
    'DoS':     ['neptune', 'smurf', 'pod', 'teardrop', 'land'],
    'Probe':   ['ipsweep', 'portsweep', 'nmap', 'satan'],
    'R2L':     ['ftp_write', 'guess_passwd', 'imap'],
    'U2R':     ['buffer_overflow', 'loadmodule', 'rootkit'],
}
evaluator.evaluate(y_true, y_pred, attack_families=attack_families)
```

### 5.3 Threshold Optimization

`NIDSEvaluator.optimize_threshold()` scans the PR curve to find the decision threshold that maximises F2-score subject to `FPR ≤ 5%`:

```
For each threshold t in PR curve:
    FPR(t) = FP(t) / total_negatives
    If FPR(t) ≤ max_fpr:
        score(t) = F2(Precision(t), Recall(t))
optimal_t = argmax(score)
```

---

## 6. Configuration Reference

All training is config-driven. Key parameters in `configs/training/hardened_rf.yaml`:

```yaml
experiment_name: nids-hybrid-v2

dataset:
  name: nsl-kdd # or: unsw-nb15, cic-ids2017
  config: configs/datasets/nsl_kdd.yaml
  test_size: 0.3
  random_state: 42
  stratify: true

preprocessing:
  apply_smote: false # BRF handles imbalance natively

feature_selection:
  method: combined # importance | rfe | shap | mutual_info | combined
  n_features: 20

models:
  use_stacking: true # Enable StackingEnsemble (Tier 1)
  use_vae: true # Enable FusionAnomalyDetector (Tier 2)
  tier1:
    config: configs/models/stacking.yaml
    stacking_params:
      n_estimators_brf: 200
      lgbm_n_estimators: 300
      lgbm_learning_rate: 0.05
      svc_C: 10.0
      cv_folds: 5
  tier2:
    config: configs/models/fusion_anomaly.yaml
    fusion_params:
      vae_weight: 0.5
      vae_epochs: 50
      vae_latent_dim: 16
      iforest_contamination: 0.05
      threshold_percentile: 95.0

training:
  normal_label: Normal
  cross_validate: true
  cv_folds: 5
  confidence_threshold: 0.90 # Tier 1 → Tier 2 routing threshold

evaluation:
  compute_shap: true
  attack_families:
    DoS: [neptune, smurf, pod, teardrop]
    Probe: [ipsweep, portsweep, nmap, satan]

tracking:
  use_mlflow: true
  mlflow_uri: mlruns
  experiment_name: nids-hybrid-v2

output:
  base_dir: experiments/runs
  save_models: true
  save_metrics: true
```

---

## 7. Technology Stack

### Core ML

| Package            | Version | Role                                 |
| ------------------ | ------- | ------------------------------------ |
| `scikit-learn`     | ≥ 1.3   | RF, SVC, RFE, preprocessing, metrics |
| `imbalanced-learn` | ≥ 0.11  | `BalancedRandomForestClassifier`     |
| `lightgbm`         | ≥ 4.3   | LightGBM base learner                |
| `torch` (PyTorch)  | ≥ 2.2   | VAE encoder/decoder                  |
| `shap`             | ≥ 0.41  | SHAP explainability                  |
| `scipy`            | ≥ 1.7   | Statistical utilities                |

### Data & Drift

| Package  | Version | Role                                   |
| -------- | ------- | -------------------------------------- |
| `numpy`  | ≥ 1.21  | Array operations                       |
| `pandas` | ≥ 1.3   | DataFrame I/O                          |
| `river`  | ≥ 0.21  | ADWIN drift detection, online learning |

### MLOps

| Package  | Version | Role                                 |
| -------- | ------- | ------------------------------------ |
| `mlflow` | ≥ 2.10  | Experiment tracking + model registry |
| `dvc`    | ≥ 3.50  | Dataset versioning (S3 remote)       |
| `joblib` | ≥ 1.1   | Model serialization                  |

### API & Deployment

| Package    | Version | Role                            |
| ---------- | ------- | ------------------------------- |
| `fastapi`  | ≥ 0.110 | Inference REST API + WebSocket  |
| `uvicorn`  | ≥ 0.29  | ASGI server                     |
| `httpx`    | ≥ 0.27  | Async HTTP client (tests)       |
| Docker     | latest  | Containerization                |
| Kubernetes | ≥ 1.28  | Orchestration (3 replicas, HPA) |

### Security

| Tool                             | Role                                    |
| -------------------------------- | --------------------------------------- |
| `adversarial-robustness-toolbox` | FGSM/PGD adversarial test generation    |
| `bandit`                         | Static security analysis of Python code |
| `safety`                         | Dependency vulnerability scan           |

### Testing

| Package      | Role               |
| ------------ | ------------------ |
| `pytest`     | Test runner        |
| `pytest-cov` | Coverage reporting |

---

## 8. CI/CD Pipeline

**File:** `.github/workflows/nids-mlops.yml`

```
push to main / weekly cron
         │
         ▼
┌─────────────────────┐
│ 1. lint-and-security│  ruff lint | bandit | safety
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. test             │  pytest --cov ≥ 60%
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. train-and-eval   │  python scripts/train.py
│                     │  python scripts/evaluate_gate.py
│                     │    PR-AUC ≥ 0.90
│                     │    F2     ≥ 0.90   ← GATE
│                     │    MCC    ≥ 0.85
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. build-and-push   │  docker build | push ghcr.io
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. deploy           │  kubectl set image (rolling update)
│                     │  Smoke test → rollback on failure
└─────────────────────┘
```

**Quality Gate Logic (`scripts/evaluate_gate.py`):**
Reads `metrics.json` from the latest experiment run, checks each threshold, exits 0 (pass) or 1 (fail — blocks Docker build and deployment).

---

## 9. Deployment Architecture

```
┌──────────────────────────────────────────────────────┐
│               Kubernetes Cluster                      │
│                                                      │
│  nids-production namespace                           │
│  ┌────────────────────────────┐                     │
│  │  nids-api Deployment       │  3 replicas          │
│  │  Image: ghcr.io/…/nids-api │  HPA: 3-12 pods     │
│  │  CPU: 500m req / 2000m lim │  on CPU > 70%        │
│  │  Mem: 1Gi req / 4Gi lim    │                     │
│  │                            │                     │
│  │  /api/predict  (POST)      │                     │
│  │  /api/feedback (POST)      │                     │
│  │  /api/shap/{id}(GET)       │                     │
│  │  /ws/live      (WebSocket) │                     │
│  └────────────────────────────┘                     │
│            │                                        │
│     ┌──────┴──────┐                                 │
│     │  Redis      │  Feature cache (sidecar)        │
│     └─────────────┘                                 │
└──────────────────────────────────────────────────────┘
           │
    MLflow Model Registry
    S3 / DVC Dataset Store
    TimescaleDB (alert history)
```

**Inference flow (end-to-end):**

1. Network packet arrives → CICFlowMeter converts to flow feature vector (41–78 features)
2. `NIDSPreprocessor.transform()` → scale + encode (≤ 1ms)
3. `FeatureSelector.transform()` → select top-N features
4. `HybridNIDS.predict()`:
   - Tier 1 `StackingEnsemble.predict_proba()` → confidence
   - If confidence ≥ 0.90 and prediction ≠ Normal → Alert immediately
   - Otherwise → Tier 2 `FusionAnomalyDetector.predict()`
5. If attack/anomaly: compute SHAP asynchronously (background task)
6. Push alert via WebSocket to SOC dashboard

---

## 10. Self-Improving Feedback Loop

```
SOC Analyst (dashboard)
      │
      ├── Approve alert → TP confirmed
      ├── Reject alert  → FP → relabeled as Normal
      └── Relabel       → correct class assigned
             │
             ▼
      FeedbackBuffer (feedback_buffer.json)
             │ trigger_size=500 samples
             ▼
      UncertaintyDiversityQuery
      (entropy + K-Means, budget=50)
             │
             ▼
      New labeled dataset (DVC commit)
             │
             ▼
      GitHub Actions retrain trigger
             │
             ▼
      evaluate_gate.py (PR-AUC / F2 / MCC gates)
             │ pass
             ▼
      MLflow Model Registry → Production
             │
             ▼
      Kubernetes rolling update
```

---

## 11. Key Files Reference

| File                                | Purpose                                              |
| ----------------------------------- | ---------------------------------------------------- |
| `nids/models/stacking.py`           | Stacking ensemble (BRF + LightGBM + CalibratedSVC)   |
| `nids/models/anomaly.py`            | VAE + FusionAnomalyDetector                          |
| `nids/models/hybrid.py`             | 4-tier cascade orchestrator                          |
| `nids/models/supervised.py`         | BalancedRF (legacy Tier 1)                           |
| `nids/models/unsupervised.py`       | IsolationForest (legacy Tier 2)                      |
| `nids/features/selection.py`        | SHAP / MI / Borda / RFE feature selection            |
| `nids/evaluation/metrics.py`        | NIDSEvaluator: MCC, PR-AUC, F2, AFI, family metrics  |
| `nids/drift/detector.py`            | ADWIN drift detector (Page-Hinkley fallback)         |
| `nids/active_learning/query.py`     | Uncertainty + diversity sample selection             |
| `nids/active_learning/feedback.py`  | SOC feedback buffer + retraining trigger             |
| `nids/pipelines/training.py`        | End-to-end training pipeline with MLflow             |
| `nids/pipelines/inference.py`       | Inference pipeline (preprocessor + selector + model) |
| `nids/explainability/`              | SHAP explainer integration                           |
| `scripts/train.py`                  | CLI training entry point                             |
| `scripts/evaluate_gate.py`          | CI/CD quality gate (PR-AUC/F2/MCC)                   |
| `.github/workflows/nids-mlops.yml`  | Full MLOps CI/CD (5-stage pipeline)                  |
| `configs/training/hardened_rf.yaml` | Primary training configuration                       |
| `deployment/Dockerfile`             | Production container definition                      |
