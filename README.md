# 🛡️ Network Intrusion Detection System (NIDS)

## What is This Project?

This is a **production-grade, hybrid Machine Learning system** for detecting network intrusions and cyber attacks. It combines two complementary detection approaches:

1. **Supervised Learning (Random Forest)** - Detects known attack patterns
2. **Unsupervised Learning (Isolation Forest)** - Identifies zero-day anomalies

### Key Features

✅ **Hybrid Architecture**: Two-tier cascade system for comprehensive threat detection  
✅ **Production-Ready**: Docker deployment with REST API  
✅ **Research-Grade**: Cross-dataset evaluation, statistical testing, SHAP explainability  
✅ **Security-Focused**: Optimized for high recall (minimizes missed attacks)  
✅ **Configurable**: YAML-based configuration management  
✅ **Reproducible**: Experiment tracking with versioned models  

### Supported Datasets

- **NSL-KDD**: Classic intrusion detection benchmark
- **UNSW-NB15**: Modern network traffic dataset
- **CIC-IDS2017**: Contemporary attack scenarios

---

## 📊 How This System Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Network Traffic                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Preprocessing       │
         │   • Clean data        │
         │   • Encode features   │
         │   • Scale values      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Feature Selection   │
         │   • RFE (20 features) │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   TIER 1: Random      │
         │   Forest Classifier   │
         └───────────┬───────────┘
                     │
            ┌────────┴────────┐
            │                 │
         Attack?           Normal?
            │                 │
            ▼                 ▼
    ┌──────────────┐  ┌──────────────────┐
    │ ALERT: Known │  │ TIER 2: Isolation│
    │ Attack Type  │  │ Forest (Anomaly) │
    └──────────────┘  └────────┬─────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                  Anomaly?          Normal?
                      │                 │
                      ▼                 ▼
              ┌──────────────┐  ┌──────────┐
              │ ALERT: Zero- │  │  PASS    │
              │ Day Attack   │  │          │
              └──────────────┘  └──────────┘
```

### Processing Pipeline

1. **Data Loading**: Load network traffic from CSV files
2. **Preprocessing**: 
   - Clean numerical data (remove inf/NaN)
   - Encode categorical features
   - Scale features using StandardScaler
   - Apply SMOTE for class balancing (training only)
3. **Feature Selection**: Use RFE to select top 20 most important features
4. **Training**:
   - **Tier 1**: Train Random Forest on all labeled data
   - **Tier 2**: Train Isolation Forest on normal traffic only
5. **Prediction**:
   - All traffic goes through Tier 1 first
   - If classified as attack → immediate alert
   - If classified as normal → pass to Tier 2
   - Tier 2 checks for anomalies (zero-day attacks)
6. **Evaluation**: Compute security-focused metrics (Recall, FNR, FAR)

### Why This Architecture?

- **Tier 1 (Random Forest)**: Excellent at recognizing known attack patterns with high accuracy
- **Tier 2 (Isolation Forest)**: Catches novel attacks that weren't in training data
- **Cascade Design**: Reduces false positives by filtering known attacks before anomaly detection

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/jivi001/Network-IDS-ML.git
cd Network-IDS-ML

# Create virtual environment
python -m venv nids_env

# Activate virtual environment
# Windows:
nids_env\Scripts\activate
# Linux/Mac:
source nids_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "from nids import HybridNIDS; print('✓ Installation successful!')"
```

---

## 📚 Project Structure

```
Network-IDS-ML/
├── nids/                      # Core Python package
│   ├── data/                  # Data loading & validation
│   ├── preprocessing/         # Data preprocessing
│   ├── features/              # Feature selection
│   ├── models/                # ML models (RF, iForest, Hybrid)
│   ├── evaluation/            # Metrics & testing
│   ├── explainability/        # SHAP interpretability
│   ├── pipelines/             # Training/evaluation/inference
│   └── utils/                 # Config & logging
├── configs/                   # YAML configuration files
│   ├── datasets/              # Dataset configurations
│   ├── models/                # Model hyperparameters
│   └── training/              # Training pipeline configs
├── scripts/                   # CLI entry points
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── cross_dataset_eval.py  # Cross-dataset testing
├── deployment/                # Docker deployment
│   ├── Dockerfile             # Production image
│   ├── docker-compose.yml     # Orchestration
│   └── inference_api.py       # REST API
├── data/                      # Dataset storage
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed data
│   └── interim/               # Intermediate files
├── models/                    # Trained models
│   ├── production/            # Production models
│   └── baselines/             # Baseline models
├── experiments/               # Experiment tracking
│   ├── runs/                  # Individual experiments
│   └── cross_dataset/         # Cross-dataset results
├── tests/                     # Unit & integration tests
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

---

## 📖 Documentation

- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Beginner's guide
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - How to train models
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** - Docker containerization
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Code documentation
- **[MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Upgrade from old version

---

## 🎯 Performance

| Dataset | Recall | Precision | F1-Score | Attack Detection |
|---------|--------|-----------|----------|------------------|
| NSL-KDD | 95.2% | 91.8% | 93.5% | 95.2% |
| UNSW-NB15 | 93.1% | 90.5% | 91.8% | 93.1% |

*Results from baseline configuration with default hyperparameters*

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **GitHub**: [@jivi001](https://github.com/jivi001)
- **Email**: jiviteshgd28@gmail.com
- **Issues**: [GitHub Issues](https://github.com/jivi001/Network-IDS-ML/issues)

---

## 🙏 Acknowledgments

- NSL-KDD dataset creators
- UNSW-NB15 dataset creators
- CIC-IDS2017 dataset creators
- scikit-learn community
