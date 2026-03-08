# Network Intrusion Detection System (NIDS-ML)

Production-grade, hybrid machine learning system for network intrusion and anomaly detection.

## Architecture

1. **Tier 1 (Known Attack Classification):** Supervised stacking ensemble (`BalancedRandomForestClassifier`, `LGBMClassifier`, `CalibratedClassifierCV` via SVC) with a `LogisticRegression` meta-learner.
2. **Tier 2 (Zero-Day Anomaly Detection):** Unsupervised fusion utilizing Variational Autoencoder (VAE) reconstruction error and `IsolationForest` decision paths.
3. **Tier 3 (Interpretability):** SHapley Additive exPlanations (`shap`) engine for real-time feature contribution analysis.
4. **Tier 4 (Adaptive Retraining):** ADWIN-based concept drift tracking and active learning (Shannon entropy + K-Means) for human-in-the-loop feedback and automated CI/CD retraining.

## Core Parameters

- **StackingEnsemble:** `n_estimators` = 200.
- **VAEAnomalyDetector:** `latent_dim` = 16, beta = 1.0.
- **FeatureSelector:** Borda count rank aggregation (RF Gini, SHAP, Shannon MI); selects 20-30 optimal features.
- **ADWINDriftDetector:** `delta` = 0.002.

## Performance Targets

- **MCC:** > 0.92
- **Alert Fatigue Index (AFI):** < 0.05
- **F2-Score:** > 0.95

## Usage

### 1. Installation

Requires Python 3.11+.

```bash
git clone https://github.com/jivi001/Network-IDS-ML.git
cd Network-IDS-ML
python -m venv nids_env
source nids_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Model Training

Expects `NSL-KDD` or `UNSW-NB15` datasets in `data/raw/`.

```bash
python scripts/train.py --config configs/training/hardened_rf.yaml
```

### 3. Backend API

Starts the FastAPI server (Port 8000) with endpoints: `/predict`, `/predict/batch`, `/explain`, and `/ws/alerts`.

```bash
cd deployment
docker-compose up --build
```

Alternatively, run directly:

```bash
python deployment/inference_api.py
```

### 4. SOC Dashboard (Frontend)

Starts the Next.js real-time UI (Port 3000).

```bash
cd ai-nids-platform/frontend/dashboard
npm install
npm run dev
```

### API Reference

Documentation available locally at `http://localhost:8000/docs`. Internal configurations are stored in the `configs/` directory.
