# Implementation Guide: Applying Repository Refactoring

This guide provides step-by-step instructions to transform your current Network-IDS-ML project structure into the production-grade architecture outlined in [Repository_Refactoring_Proposal.md](Repository_Refactoring_Proposal.md).

---

## Quick Start Options

### Option A: Full Refactoring (Recommended for Long-term)
Follow all 5 phases for complete transformation to production-grade structure.

**Time Required**: 7-11 days  
**Best For**: Projects intended for production deployment, academic publication, or portfolio showcase

### Option B: Minimal Refactoring (Quick Wins)
Implement only Phase 1 (core structure) and Phase 4 (documentation).

**Time Required**: 2-3 days  
**Best For**: Academic submissions with tight deadlines

### Option C: Documentation Only (Immediate Use)
Use the existing Methodology.md for your report/paper without code refactoring.

**Time Required**: 0 days (already complete)  
**Best For**: Immediate project submission requirements

---

## Phase 1: Core Directory Restructuring (Priority: HIGH)

### Step 1.1: Create New Directory Structure

Run this PowerShell script from your project root:

```powershell
# Create core package structure
New-Item -ItemType Directory -Path "nids\data" -Force
New-Item -ItemType Directory -Path "nids\preprocessing" -Force
New-Item -ItemType Directory -Path "nids\features" -Force
New-Item -ItemType Directory -Path "nids\models" -Force
New-Item -ItemType Directory -Path "nids\evaluation" -Force
New-Item -ItemType Directory -Path "nids\utils" -Force
New-Item -ItemType Directory -Path "nids\pipelines" -Force

# Create config structure
New-Item -ItemType Directory -Path "configs\datasets" -Force
New-Item -ItemType Directory -Path "configs\models" -Force

# Create experiment structure
New-Item -ItemType Directory -Path "experiments\runs" -Force
New-Item -ItemType Directory -Path "experiments\notebooks" -Force
New-Item -ItemType Directory -Path "experiments\results" -Force

# Create scripts directory
New-Item -ItemType Directory -Path "scripts" -Force

# Create tests structure
New-Item -ItemType Directory -Path "tests\unit" -Force
New-Item -ItemType Directory -Path "tests\integration" -Force
New-Item -ItemType Directory -Path "tests\fixtures" -Force

# Create deployment structure
New-Item -ItemType Directory -Path "deployment\docker" -Force
New-Item -ItemType Directory -Path "deployment\kubernetes" -Force
New-Item -ItemType Directory -Path "deployment\scripts" -Force

# Reorganize data directories
New-Item -ItemType Directory -Path "data\processed" -Force
New-Item -ItemType Directory -Path "data\interim" -Force
```

### Step 1.2: Create __init__.py Files

```powershell
# Create all __init__.py files
New-Item -ItemType File -Path "nids\__init__.py" -Force
New-Item -ItemType File -Path "nids\data\__init__.py" -Force
New-Item -ItemType File -Path "nids\preprocessing\__init__.py" -Force
New-Item -ItemType File -Path "nids\features\__init__.py" -Force
New-Item -ItemType File -Path "nids\models\__init__.py" -Force
New-Item -ItemType File -Path "nids\evaluation\__init__.py" -Force
New-Item -ItemType File -Path "nids\utils\__init__.py" -Force
New-Item -ItemType File -Path "nids\pipelines\__init__.py" -Force
New-Item -ItemType File -Path "tests\__init__.py" -Force
```

### Step 1.3: Move Existing Files

**Manual file moves** (review each before moving):

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `src/data_loader.py` | `nids/data/loaders.py` | Move & rename |
| `src/download_datasets.py` | `nids/data/downloaders.py` | Move & rename |
| `src/feature_selection.py` | `nids/features/selection.py` | Move |
| `src/hybrid_system.py` | `nids/models/hybrid.py` | Move |
| `main.py` | `scripts/run_experiment.py` | Move & refactor |

**Files requiring splitting**:

1. **src/preprocessing.py** → Split into:
   - `nids/preprocessing/cleaners.py` (handle_inf, impute_missing)
   - `nids/preprocessing/encoders.py` (LabelEncoder wrapper)
   - `nids/preprocessing/scalers.py` (StandardScaler wrapper)
   - `nids/preprocessing/balancers.py` (SMOTE logic)

2. **src/models.py** → Split into:
   - `nids/models/supervised.py` (RandomForest, SVM)
   - `nids/models/unsupervised.py` (IsolationForest)

3. **src/evaluation.py** → Split into:
   - `nids/evaluation/metrics.py` (metric calculations)
   - `nids/evaluation/visualizations.py` (plotting functions)

### Step 1.4: Update Import Statements

After moving files, update all imports. Example:

**Before**:
```python
from src.preprocessing import NIDSPreprocessor
from src.models import SupervisedModel
from src.hybrid_system import HybridNIDS
```

**After**:
```python
from nids.preprocessing.cleaners import NIDSPreprocessor
from nids.models.supervised import RandomForestModel
from nids.models.hybrid import HybridNIDS
```

---

## Phase 2: Configuration Externalization (Priority: MEDIUM)

### Step 2.1: Create Configuration Files

Create `configs/models/random_forest.yaml`:

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

Create `configs/models/isolation_forest.yaml`:

```yaml
model_type: IsolationForest
hyperparameters:
  n_estimators: 100
  contamination: 0.1
  max_samples: 256
  random_state: 42
```

Create `configs/preprocessing.yaml`:

```yaml
preprocessing:
  scaling:
    method: standard  # standard, minmax, robust
  encoding:
    method: label  # label, onehot
  smote:
    k_neighbors: 5
    sampling_strategy: auto
  cleaning:
    inf_strategy: max  # max, drop
    nan_strategy: median  # median, mean, drop
```

### Step 2.2: Create Config Loader Utility

Create `nids/utils/config.py`:

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config_path(config_name: str, config_type: str = 'models') -> Path:
    """Get full path to config file."""
    base_path = Path(__file__).parent.parent.parent / 'configs'
    return base_path / config_type / f"{config_name}.yaml"
```

### Step 2.3: Update Model Initialization

**Before** (hardcoded):
```python
nids = HybridNIDS(
    rf_params={'n_estimators': 100, 'max_depth': 20},
    iforest_params={'contamination': 0.1}
)
```

**After** (config-driven):
```python
from nids.utils.config import load_config, get_config_path

rf_config = load_config(get_config_path('random_forest'))
if_config = load_config(get_config_path('isolation_forest'))

nids = HybridNIDS(
    rf_params=rf_config['hyperparameters'],
    iforest_params=if_config['hyperparameters']
)
```

---

## Phase 3: Pipeline Development (Priority: MEDIUM)

### Step 3.1: Create Training Pipeline

Create `nids/pipelines/training.py`:

```python
from pathlib import Path
from typing import Dict, Optional
import joblib

from nids.data.loaders import DataLoader
from nids.preprocessing.cleaners import NIDSPreprocessor
from nids.preprocessing.balancers import SMOTEBalancer
from nids.features.selection import RFFeatureSelector
from nids.models.hybrid import HybridNIDS
from nids.utils.config import load_config

class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
    def run(self, train_data_path: str, output_dir: str):
        """Execute full training pipeline."""
        # Load data
        loader = DataLoader()
        X_train, y_train = loader.load(train_data_path)
        
        # Preprocess
        preprocessor = NIDSPreprocessor()
        X_processed = preprocessor.fit_transform(X_train)
        
        # Balance classes
        balancer = SMOTEBalancer()
        X_balanced, y_balanced = balancer.fit_resample(X_processed, y_train)
        
        # Feature selection
        selector = RFFeatureSelector(n_features=20)
        X_selected = selector.fit_transform(X_balanced, y_balanced)
        
        # Train hybrid system
        nids = HybridNIDS(
            rf_params=self.config['rf_params'],
            iforest_params=self.config['iforest_params']
        )
        nids.train(X_selected, y_balanced)
        
        # Save artifacts
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        nids.save(
            str(output_path / 'tier1_rf.pkl'),
            str(output_path / 'tier2_iforest.pkl')
        )
        joblib.dump(preprocessor, output_path / 'preprocessor.pkl')
        joblib.dump(selector, output_path / 'feature_selector.pkl')
        
        print(f"✓ Training complete. Models saved to {output_dir}")
```

### Step 3.2: Create Experiment Runner Script

Create `scripts/run_experiment.py`:

```python
import argparse
from pathlib import Path
from nids.pipelines.training import TrainingPipeline

def main():
    parser = argparse.ArgumentParser(description='Run NIDS training experiment')
    parser.add_argument('--config', required=True, help='Path to training config')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(config_path=args.config)
    pipeline.run(train_data_path=args.data, output_dir=args.output)

if __name__ == '__main__':
    main()
```

**Usage**:
```bash
python scripts/run_experiment.py \
    --config configs/training.yaml \
    --data data/processed/nsl_kdd_train.pkl \
    --output experiments/runs/exp_001_baseline
```

---

## Phase 4: Documentation Updates (Priority: HIGH)

### Step 4.1: Update README.md

Add references to new documentation:

```markdown
## 📚 Documentation

- **[Methodology](docs/Methodology.md)**: Complete research methodology for academic papers
- **[Repository Structure](docs/Repository_Refactoring_Proposal.md)**: Production-grade project organization
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)**: Step-by-step refactoring instructions
```

### Step 4.2: Create Dataset Guide

Create `docs/dataset_guide.md` with instructions for downloading NSL-KDD, UNSW-NB15, and CIC-IDS2017.

### Step 4.3: Create API Reference

Create `docs/api_reference.md` documenting key classes and functions.

---

## Phase 5: Deployment Preparation (Priority: LOW)

### Step 5.1: Create Dockerfile

Create `deployment/docker/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nids/ ./nids/
COPY models/production/ ./models/production/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["python", "-m", "nids.api.server"]
```

### Step 5.2: Create setup.py

Create `setup.py` in project root:

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
        'pyyaml>=5.4.0',
        'joblib>=1.0.0'
    ],
    python_requires='>=3.8',
    author='Your Name',
    description='Hybrid ML-based Network Intrusion Detection System',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
```

**Install in editable mode**:
```bash
pip install -e .
```

---

## Testing Your Refactored Structure

### Verify Directory Structure

```powershell
tree /F /A
```

Expected output should match the structure in `Repository_Refactoring_Proposal.md`.

### Test Imports

Create `tests/test_imports.py`:

```python
def test_imports():
    """Verify all modules can be imported."""
    from nids.data.loaders import DataLoader
    from nids.preprocessing.cleaners import NIDSPreprocessor
    from nids.models.hybrid import HybridNIDS
    from nids.evaluation.metrics import calculate_metrics
    print("✓ All imports successful")

if __name__ == '__main__':
    test_imports()
```

Run:
```bash
python tests/test_imports.py
```

### Run End-to-End Test

```bash
python scripts/run_experiment.py \
    --config configs/training.yaml \
    --data data/processed/nsl_kdd_train.pkl \
    --output experiments/runs/test_run
```

---

## Troubleshooting

### Import Errors After Refactoring

**Problem**: `ModuleNotFoundError: No module named 'nids'`

**Solution**: Install package in editable mode:
```bash
pip install -e .
```

### Config File Not Found

**Problem**: `FileNotFoundError: configs/models/random_forest.yaml`

**Solution**: Ensure you're running scripts from project root, or use absolute paths in config loader.

### Circular Import Issues

**Problem**: `ImportError: cannot import name 'X' from partially initialized module`

**Solution**: Review import order; move imports inside functions if necessary.

---

## Recommended Implementation Order

For academic project submission:

1. **Week 1**: Phase 1 (Core Restructuring) + Phase 4 (Documentation)
2. **Week 2**: Phase 2 (Configuration) + Phase 3 (Pipelines)
3. **Week 3**: Phase 5 (Deployment) + Testing & Validation

For immediate use (paper/report):

- Use `docs/Methodology.md` directly in your report
- Reference current codebase in GitHub
- Plan refactoring for post-submission improvements

---

## Additional Resources

- **Methodology Document**: [docs/Methodology.md](Methodology.md)
- **Refactoring Proposal**: [docs/Repository_Refactoring_Proposal.md](Repository_Refactoring_Proposal.md)
- **Current README**: [../README.md](../README.md)

---

**Questions or Issues?**

If you encounter problems during refactoring, refer to the detailed component placement guide in `Repository_Refactoring_Proposal.md` Section 4.
