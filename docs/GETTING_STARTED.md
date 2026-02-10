# 🚀 Getting Started with NIDS

Welcome! This guide will help you get up and running with the Network Intrusion Detection System in **under 10 minutes**.

---

## What You'll Learn

1. Install the system
2. Download a sample dataset
3. Train your first model
4. Make predictions
5. Deploy with Docker

---

## Step 1: Installation (2 minutes)

### Clone Repository

```bash
git clone https://github.com/jivi001/Network-IDS-ML.git
cd Network-IDS-ML
```

### Create Virtual Environment

**Windows:**
```bash
python -m venv nids_env
nids_env\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv nids_env
source nids_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "from nids import HybridNIDS; print('✓ Installation successful!')"
```

---

## Step 2: Get Sample Dataset (1 minute)

### Download NSL-KDD Dataset

```bash
# Create data directory
mkdir -p data/raw

# Download NSL-KDD (example URLs - replace with actual sources)
# Training set
wget https://example.com/nsl_kdd_train.csv -O data/raw/nsl_kdd_train.csv

# Test set
wget https://example.com/nsl_kdd_test.csv -O data/raw/nsl_kdd_test.csv
```

**Note**: Replace URLs with actual dataset sources. You can find NSL-KDD on Kaggle or UCI ML Repository.

### Verify Dataset

```bash
# Check files exist
ls data/raw/

# Output should show:
# nsl_kdd_train.csv
# nsl_kdd_test.csv
```

---

## Step 3: Train Your First Model (5 minutes)

### Run Training

```bash
python scripts/train.py --config configs/training/default.yaml
```

### What Happens During Training?

1. **Data Loading**: Loads NSL-KDD dataset
2. **Preprocessing**: Cleans and scales features
3. **SMOTE**: Balances classes
4. **Feature Selection**: Selects top 20 features using RFE
5. **Training**: 
   - Tier 1: Random Forest on all data
   - Tier 2: Isolation Forest on normal traffic
6. **Evaluation**: Computes metrics and saves plots

### Expected Output

```
Starting experiment: hybrid_nids_baseline_20260210_215000

Loading data...
Train: (125973, 41), Test: (22544, 41)

Preprocessing...
[OK] Preprocessing complete

SMOTE...
[OK] SMOTE complete: 126000 samples

Feature selection...
[OK] Selected 20 features

Training Tier 1 (Random Forest)...
[OK] Tier 1 trained

Training Tier 2 (Isolation Forest)...
[OK] Tier 2 trained

Evaluating...
Accuracy:  0.9523
Recall:    0.9523
Precision: 0.9180
F1-Score:  0.9348

✓ Experiment complete!
Results saved to: experiments/runs/hybrid_nids_baseline_20260210_215000
```

### Check Results

```bash
# View metrics
cat experiments/runs/hybrid_nids_baseline_20260210_215000/metrics.json

# View plots
ls experiments/runs/hybrid_nids_baseline_20260210_215000/plots/
# confusion_matrix.png
# pr_curve.png
# shap_summary.png
```

---

## Step 4: Make Predictions (1 minute)

### Test Prediction

Create a test script `test_prediction.py`:

```python
import sys
sys.path.insert(0, 'e:/NIDS ML')

from nids.pipelines import InferencePipeline
import numpy as np

# Load trained model
pipeline = InferencePipeline(model_version='v1.0.0')

# Create sample traffic (20 features after feature selection)
sample_traffic = np.random.randn(20)

# Predict
result = pipeline.predict_single(sample_traffic)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Tier used: {result['tier_used']}")
print(f"Anomaly score: {result['anomaly_score']:.4f}")
```

**Note**: First copy your trained model to production:
```bash
mkdir -p models/production/v1.0.0
cp experiments/runs/YOUR_EXPERIMENT_ID/models/* models/production/v1.0.0/
```

---

## Step 5: Deploy with Docker (2 minutes)

### Build Docker Image

```bash
docker build -t nids:v1.0.0 -f deployment/Dockerfile .
```

### Run Container

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.7, 0.9, 1.1, 0.4,
                 0.6, 1.3, 0.2, 1.8, 0.5, 1.0, 0.8, 1.4, 0.6, 0.9]
  }'
```

### Expected Response

```json
{
  "prediction": "Normal",
  "confidence": 0.95,
  "tier_used": 2,
  "anomaly_score": 0.12
}
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Installed the NIDS system
- ✅ Trained a hybrid ML model
- ✅ Made predictions
- ✅ Deployed with Docker

---

## 📚 Next Steps

### Learn More

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Advanced training techniques
- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Docker deployment details
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API_REFERENCE.md](API_REFERENCE.md)** - Code documentation

### Try Advanced Features

1. **Cross-Dataset Evaluation**:
   ```bash
   python scripts/cross_dataset_eval.py \
     --source-model experiments/runs/YOUR_EXPERIMENT/models \
     --source-dataset nsl_kdd \
     --target-dataset unsw_nb15 \
     --target-data data/raw/unsw_nb15_test.csv
   ```

2. **Hyperparameter Tuning**:
   - Edit `configs/models/random_forest.yaml`
   - Increase `n_estimators` for better performance
   - Adjust `max_depth` for model complexity

3. **SHAP Explainability**:
   - Check `experiments/runs/YOUR_EXPERIMENT/plots/shap_summary.png`
   - Understand which features drive predictions

---

## 🐛 Troubleshooting

### Issue: Module not found

**Solution**: Make sure you installed the package:
```bash
pip install -e .
```

### Issue: Dataset not found

**Solution**: Check file paths in `configs/datasets/nsl_kdd.yaml`:
```yaml
paths:
  train: data/raw/nsl_kdd_train.csv  # Update this
  test: data/raw/nsl_kdd_test.csv    # Update this
```

### Issue: Out of memory

**Solution**: Reduce dataset size for testing:
```bash
# Use only first 10000 rows
head -n 10000 data/raw/nsl_kdd_train.csv > data/raw/nsl_kdd_train_small.csv
```

### Issue: Docker build fails

**Solution**: Ensure model files exist:
```bash
ls models/production/v1.0.0/
# Should show: tier1_rf.pkl, tier2_iforest.pkl, preprocessor.pkl, feature_selector.pkl
```

---

## 💡 Quick Tips

1. **Start Small**: Use a subset of data for initial testing
2. **Monitor Resources**: Training can be memory-intensive
3. **Save Experiments**: Each training run is saved with timestamp
4. **Check Logs**: Look in `logs/` directory for debugging
5. **Use Configs**: Modify YAML files instead of code

---

## 📧 Need Help?

- **Documentation**: Check `docs/` directory
- **Issues**: Open a GitHub issue
- **Examples**: See `notebooks/` for Jupyter examples

---

**Ready to dive deeper?** Check out the [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced techniques!
