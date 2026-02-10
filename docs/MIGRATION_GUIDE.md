# Migration Guide: Old Structure → Production-Grade NIDS

This guide helps you migrate from the old monolithic structure to the new production-grade package.

---

## Overview of Changes

### Import Path Changes

**Before:**
```python
from src.data_loader import DataLoader
from src.preprocessing import NIDSPreprocessor
from src.feature_selection import FeatureSelector
from src.models import SupervisedModel, UnsupervisedModel
from src.hybrid_system import HybridNIDS
from src.evaluation import NIDSEvaluator
```

**After:**
```python
from nids.data import DataLoader
from nids.preprocessing import NIDSPreprocessor
from nids.features import FeatureSelector
from nids.models import SupervisedModel, UnsupervisedModel, HybridNIDS
from nids.evaluation import NIDSEvaluator
```

### Configuration Changes

**Before (Hardcoded):**
```python
hybrid_nids = HybridNIDS(
    rf_params={'n_estimators': 200, 'max_depth': 20},
    iforest_params={'contamination': 0.05}
)
```

**After (Config-Driven):**
```python
from nids.utils.config import load_config

config = load_config('configs/training/default.yaml')
tier1_config = load_config(config['models']['tier1']['config'])
tier2_config = load_config(config['models']['tier2']['config'])

hybrid_nids = HybridNIDS(
    rf_params=tier1_config['hyperparameters'],
    iforest_params=tier2_config['hyperparameters']
)
```

### Model Saving Changes

**Before:**
```python
hybrid_nids.save('models/rf_model.pkl', 'models/iforest_model.pkl')
```

**After:**
```python
# Versioned model storage
models_dir = Path('models/production/v1.0.0')
models_dir.mkdir(parents=True, exist_ok=True)
hybrid_nids.save(
    str(models_dir / 'tier1_rf.pkl'),
    str(models_dir / 'tier2_iforest.pkl')
)
```

---

## Step-by-Step Migration

### Step 1: Update Imports

Use find-and-replace to update import statements:

```bash
# In all Python files
from src.data_loader → from nids.data.loaders
from src.preprocessing → from nids.preprocessing.preprocessor
from src.feature_selection → from nids.features.selection
from src.models → from nids.models.supervised (or unsupervised)
from src.hybrid_system → from nids.models.hybrid
from src.evaluation → from nids.evaluation.metrics
```

### Step 2: Migrate to Configuration Files

1. Create `configs/training/my_experiment.yaml`
2. Move hardcoded hyperparameters to YAML
3. Update training code to load config:

```python
from nids.pipelines import TrainingPipeline

pipeline = TrainingPipeline('configs/training/my_experiment.yaml')
experiment_id, metrics = pipeline.run()
```

### Step 3: Update Model Paths

Replace direct model saving with versioned storage:

```python
import joblib
from pathlib import Path

# Create versioned directory
version = 'v1.0.0'
model_dir = Path(f'models/production/{version}')
model_dir.mkdir(parents=True, exist_ok=True)

# Save models
hybrid_nids.save(
    str(model_dir / 'tier1_rf.pkl'),
    str(model_dir / 'tier2_iforest.pkl')
)

# Save preprocessor and selector
joblib.dump(preprocessor, model_dir / 'preprocessor.pkl')
joblib.dump(selector, model_dir / 'feature_selector.pkl')
```

### Step 4: Use Pipelines

Replace custom training scripts with pipelines:

**Old Approach:**
```python
# main.py with 300+ lines of procedural code
```

**New Approach:**
```python
# Use training pipeline
from nids.pipelines import TrainingPipeline

pipeline = TrainingPipeline('configs/training/default.yaml')
experiment_id, metrics = pipeline.run()
```

---

## Backward Compatibility

The old `src/` modules are still present but deprecated. To ensure compatibility:

1. Keep `src/` directory temporarily
2. Update code incrementally
3. Test with new imports
4. Remove `src/` once migration is complete

---

## Testing Your Migration

```bash
# Run tests to verify functionality
pytest tests/unit/test_preprocessing.py -v
pytest tests/integration/test_training_pipeline.py -v

# Compare results
python scripts/train.py --config configs/training/default.yaml
# Check that metrics match previous results
```

---

## Common Issues

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'nids'`

**Solution:** Install package in development mode:
```bash
pip install -e .
```

### Issue 2: Config File Not Found

**Error:** `FileNotFoundError: Config file not found`

**Solution:** Use absolute paths or ensure working directory is project root:
```python
from pathlib import Path
config_path = Path(__file__).parent / 'configs' / 'training' / 'default.yaml'
```

### Issue 3: Model Loading Fails

**Error:** `FileNotFoundError: Model not found`

**Solution:** Update model paths to new versioned structure:
```python
model_dir = Path('models/production/v1.0.0')
hybrid_nids.load(
    str(model_dir / 'tier1_rf.pkl'),
    str(model_dir / 'tier2_iforest.pkl'),
    normal_label='Normal'
)
```

---

## Rollback Plan

If migration fails, you can rollback:

1. Revert import changes
2. Use old `src/` modules
3. Keep old `main.py` as fallback

The new structure is designed to coexist with the old one during transition.

---

## Next Steps

After migration:

1. ✅ Delete old `src/` directory
2. ✅ Update documentation
3. ✅ Train new baseline model
4. ✅ Run cross-dataset evaluation
5. ✅ Deploy with Docker

---

## Support

For migration assistance, open an issue on GitHub or contact the maintainers.
