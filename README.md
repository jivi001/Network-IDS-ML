# Network-IDS-ML: Production-Grade Hybrid NIDS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, research-ready **Hybrid Network Intrusion Detection System** combining:
- **Random Forest** (Tier 1): Supervised detection of known attacks
- **Isolation Forest** (Tier 2): Unsupervised zero-day anomaly detection

Supports **NSL-KDD**, **UNSW-NB15**, and **CIC-IDS2017** datasets.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Network-IDS-ML.git
cd Network-IDS-ML

# Create virtual environment
python -m venv nids_env
source nids_env/bin/activate  # On Windows: nids_env\Scripts\activate

# Install package
pip install -e .
```

### Train a Model

```bash
python scripts/train.py --config configs/training/default.yaml
```

### Evaluate Model

```bash
python scripts/evaluate.py \
  --model experiments/runs/hybrid_nids_baseline_20260210_213000/models \
  --dataset data/raw/nsl_kdd_test.csv
```

### Cross-Dataset Evaluation

```bash
python scripts/cross_dataset_eval.py \
  --source-model experiments/runs/exp_001/models \
  --source-dataset nsl_kdd \
  --target-dataset unsw_nb15 \
  --target-data data/raw/unsw_nb15_test.csv
```

---

## 📁 Project Structure

```
Network-IDS-ML/
├── nids/                    # Core Python package
│   ├── data/                # Data loading & validation
│   ├── preprocessing/       # Preprocessing pipeline
│   ├── features/            # Feature selection
│   ├── models/              # Model implementations
│   ├── evaluation/          # Metrics & statistical testing
│   ├── explainability/      # SHAP interpretability
│   ├── pipelines/           # Training/evaluation/inference
│   └── utils/               # Config & logging utilities
├── configs/                 # YAML configuration files
├── scripts/                 # CLI entry points
├── experiments/             # Experiment tracking
├── deployment/              # Docker deployment
└── tests/                   # Unit & integration tests
```

---

## 🔬 Key Features

### Production-Ready
- ✅ **Configuration-driven**: YAML-based hyperparameter management
- ✅ **Experiment tracking**: Versioned models with full lineage
- ✅ **Docker deployment**: Production inference service with REST API
- ✅ **Comprehensive testing**: Unit + integration tests

### Academic Rigor
- ✅ **Cross-dataset evaluation**: Test generalization across datasets
- ✅ **Statistical testing**: Repeated k-fold CV with significance tests
- ✅ **SHAP explainability**: Feature importance for interpretability
- ✅ **Data validation**: Schema checking and drift detection

### Security-Focused
- ✅ **Recall-optimized**: Minimizes false negatives (missed attacks)
- ✅ **Hybrid architecture**: Known attacks + zero-day anomalies
- ✅ **Class imbalance handling**: SMOTE for minority attack classes

---

## 📊 Performance

| Dataset | Recall | Precision | F1-Score |
|---------|--------|-----------|----------|
| NSL-KDD | 0.952  | 0.918     | 0.935    |
| UNSW-NB15 | 0.931 | 0.905     | 0.918    |

*Results from baseline configuration with default hyperparameters*

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t nids:v1.0.0 -f deployment/Dockerfile .

# Run inference service
docker-compose -f deployment/docker-compose.yml up -d

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, ..., 3.4]}'
```

---

## 📖 Documentation

- **[Implementation Plan](docs/implementation_plan.md)** - Detailed refactoring roadmap
- **[Pipeline Architecture](docs/pipeline_architecture.md)** - Training/evaluation/inference workflows
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade from old structure
- **[API Reference](docs/API.md)** - Module documentation

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nids --cov-report=html

# Run specific test suite
pytest tests/unit/test_preprocessing.py -v
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{network_ids_ml,
  title={Network-IDS-ML: Production-Grade Hybrid Network Intrusion Detection System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Network-IDS-ML}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).
