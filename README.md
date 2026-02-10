# Hybrid Network Intrusion Detection System (NIDS)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Machine Learning-based Network Intrusion Detection System** implementing a hybrid two-tier cascade architecture for comprehensive threat detection.

## 🎯 Overview

This system combines the strengths of supervised and unsupervised learning to detect both known attacks and zero-day anomalies:

- **Tier 1 (Supervised)**: Random Forest classifier for known attack patterns
- **Tier 2 (Unsupervised)**: Isolation Forest for zero-day anomaly detection
- **Cascade Logic**: Tier 2 only processes traffic classified as "Normal" by Tier 1

## 🏗️ Architecture

```
Network Traffic → Preprocessing → Feature Selection → Tier 1 (RF)
                                                          ↓
                                                    [Attack] → Alert
                                                          ↓
                                                    [Normal] → Tier 2 (iForest)
                                                                  ↓
                                                            [Anomaly] → Alert
                                                                  ↓
                                                            [Normal] → Pass
```

## 🚀 Features

- **Hybrid Detection**: Combines supervised (Random Forest) and unsupervised (Isolation Forest) learning
- **SMOTE Balancing**: Handles class imbalance in training data
- **Security-Focused**: Optimized for Recall (minimizes missed attacks)
- **Modular Design**: Clean separation of concerns across 7 core modules
- **Dataset Support**: Compatible with NSL-KDD, UNSW-NB15, CIC-IDS2017
- **Visualization**: Confusion matrix, Precision-Recall curves, feature importance

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Methodology](docs/Methodology.md)**: Complete research methodology suitable for academic papers and final-year projects
  - Problem formulation (ML vs signature-based IDS)
  - Dataset justification (NSL-KDD, UNSW-NB15, CIC-IDS2017)
  - Preprocessing pipeline (cleaning, encoding, scaling, SMOTE)
  - Model architecture (Random Forest, SVM, Isolation Forest)
  - Hybrid cascade workflow
  - Evaluation metrics and limitations

- **[Repository Refactoring Proposal](docs/Repository_Refactoring_Proposal.md)**: Production-grade project structure
  - Modular architecture design
  - Configuration management
  - Experiment tracking
  - Deployment artifacts
  - SOC/industry alignment

- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)**: Step-by-step refactoring instructions
  - Phase-by-phase migration plan
  - PowerShell commands for Windows
  - Code examples and troubleshooting
  - Testing procedures

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/jivi001/Network-IDS-ML.git
cd Network-IDS-ML

# Create virtual environment (recommended)
python -m venv nids_env
source nids_env/bin/activate  # On Windows: nids_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Quick Start

### Run with Synthetic Data

```bash
python main.py
```

This will:
1. Generate 5000 synthetic network traffic samples
2. Apply preprocessing (cleaning, encoding, scaling)
3. Balance classes using SMOTE
4. Train the hybrid NIDS
5. Evaluate performance
6. Save visualizations to `logs/`

### Use with Real Datasets

```python
from src.data_loader import DataLoader
from src.preprocessing import NIDSPreprocessor
from src.hybrid_system import HybridNIDS

# Load dataset
loader = DataLoader(dataset_type='cic-ids2017')
df = loader.load_csv('data/raw/dataset.csv')
X, y = loader.split_features_labels(df)

# Preprocess
preprocessor = NIDSPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_train_balanced, y_train_balanced = preprocessor.apply_smote(
    X_train_processed, y_train
)

# Train
nids = HybridNIDS(
    rf_params={'n_estimators': 100, 'max_depth': 20},
    iforest_params={'contamination': 0.1}
)
nids.train(X_train_balanced, y_train_balanced, normal_label='Normal')

# Predict
predictions, tier_flags = nids.predict(X_test_processed)
```

## 📂 Project Structure

```
Network-IDS-ML/
├── data/               # Dataset storage (gitignored)
├── logs/               # Evaluation outputs (confusion matrix, PR curves)
├── models/             # Saved model artifacts
├── notebooks/          # Jupyter notebooks for analysis
├── src/
│   ├── data_loader.py       # CSV loading with schema detection
│   ├── preprocessing.py     # Cleaning, encoding, scaling, SMOTE
│   ├── feature_selection.py # RF importance-based selection
│   ├── models.py            # RF and Isolation Forest wrappers
│   ├── hybrid_system.py     # Cascade architecture implementation
│   └── evaluation.py        # Security-focused metrics
├── main.py             # End-to-end pipeline demo
└── requirements.txt    # Python dependencies
```

## 🔬 Key Components

### Data Preprocessing
- **Cleaning**: Replaces `inf` with column max, imputes `NaN` with median
- **Encoding**: Label encoding for categorical features
- **Scaling**: StandardScaler (Z-score normalization)
- **SMOTE**: Applied **only** to training data to prevent data leakage

### Hybrid System
- **Tier 1**: Random Forest trained on all labeled data (balanced)
- **Tier 2**: Isolation Forest trained **only** on normal samples
- **Cascade**: Tier 2 processes only RF-Normal predictions

### Evaluation
- **Metrics**: Precision, Recall, F1-Score, Confusion Matrix
- **Focus**: Recall prioritized (minimize False Negatives)
- **Visualizations**: Confusion matrix heatmap, Precision-Recall curve

## 📊 Performance

On synthetic test data (1500 samples):
- **Weighted Recall**: ~95% (attack detection rate)
- **Weighted Precision**: ~92%
- **F1-Score**: ~93%
- **Tier 2 Activation**: 70% of samples (cascade working correctly)
- **Zero-Day Detections**: ~105 anomalies flagged

## 🛠️ Supported Datasets

- **NSL-KDD**: Legacy benchmark (41 features)
- **UNSW-NB15**: Modern hybrid dataset (49 features, 9 attack families)
- **CIC-IDS2017**: State-of-the-art (80+ features, modern attack vectors)

Place datasets in `data/raw/` directory.

## 📈 Extending the System

### Add Custom Preprocessing
```python
from src.preprocessing import NIDSPreprocessor

class CustomPreprocessor(NIDSPreprocessor):
    def custom_transform(self, X):
        # Your custom logic
        return X
```

### Tune Hyperparameters
```python
nids = HybridNIDS(
    rf_params={
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5
    },
    iforest_params={
        'contamination': 0.05,
        'n_estimators': 150
    }
)
```

## 🔐 Security Considerations

- **Recall Priority**: System optimized to minimize missed attacks (False Negatives)
- **Alert Fatigue**: Precision balanced to avoid overwhelming analysts
- **Zero-Day Detection**: Tier 2 provides defense against unknown threats
- **No Data Leakage**: SMOTE strictly applied to training set only

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research based on comprehensive analysis of ML-based NIDS architectures
- Dataset providers: NSL-KDD, UNSW-NB15, CIC-IDS2017
- Built with scikit-learn, imbalanced-learn, pandas, matplotlib

## 📧 Contact

For questions or collaboration:
- GitHub: [@jivi001](https://github.com/jivi001)
- Repository: [Network-IDS-ML](https://github.com/jivi001/Network-IDS-ML)

---

**Note**: This is a research/educational implementation. For production deployment, additional hardening, monitoring, and integration with existing security infrastructure is recommended.
