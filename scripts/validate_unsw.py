"""
Validate hardened NIDS on UNSW-NB15.
Reuses the same pipeline as NSL-KDD validation but with UNSW data.
"""
import sys, json, time, numpy as np
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.data import DataLoader
from nids.preprocessing import NIDSPreprocessor
from nids.features.selection import FeatureSelector
from nids.models import HybridNIDS
from nids.evaluation import NIDSEvaluator
from nids.utils.config import load_config
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

OUTPUT_DIR = Path("experiments/unsw_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load UNSW-NB15 ─────────────────────────────────────────────
loader = DataLoader(dataset_type='unsw_nb15')
df_train = loader.load_csv('data/raw/unsw_nb15_train.csv')
df_test  = loader.load_csv('data/raw/unsw_nb15_test.csv')
X_train, y_train = loader.split_features_labels(df_train)
X_test, y_test   = loader.split_features_labels(df_test)
# Fill NaN from numeric coercion of raw CSVs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
print(f"UNSW-NB15: Train={X_train.shape}, Test={X_test.shape}")
print(f"  Train labels: {dict(y_train.value_counts())}")
print(f"  Test labels:  {dict(y_test.value_counts())}")

# ── Preprocess ──────────────────────────────────────────────────
prep = NIDSPreprocessor(random_state=42)
X_tr = prep.fit_transform(X_train)
X_te = prep.transform(X_test)

# ── Leak-free CV ────────────────────────────────────────────────
rf_config = load_config('configs/models/random_forest_hardened.yaml')
rf_params = rf_config['hyperparameters']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_cv, f1_cv, f2_cv = [], [], []

print("\n5-fold CV (leak-free: SMOTE + feat select inside folds):")
for i, (tr_idx, val_idx) in enumerate(cv.split(X_tr, y_train)):
    Xtr, Xval = X_tr[tr_idx], X_tr[val_idx]
    ytr = y_train.values[tr_idx]
    yval = y_train.values[val_idx]

    sm = SMOTE(random_state=42, k_neighbors=5)
    Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)

    # Quick feature selection inside fold
    mini = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    mini.fit(Xtr_b, ytr_b)
    top = np.argsort(mini.feature_importances_)[-20:]
    Xtr_s, Xval_s = Xtr_b[:, top], Xval[:, top]

    m = RandomForestClassifier(**rf_params)
    m.fit(Xtr_s, ytr_b)
    yp = m.predict(Xval_s)
    r = recall_score(yval, yp, average='weighted', zero_division=0)
    f1 = f1_score(yval, yp, average='weighted', zero_division=0)
    f2 = fbeta_score(yval, yp, beta=2, average='weighted', zero_division=0)
    recall_cv.append(r); f1_cv.append(f1); f2_cv.append(f2)
    print(f"  Fold {i+1}: Recall={r:.4f} F1={f1:.4f} F2={f2:.4f}")

print(f"  Mean Recall: {np.mean(recall_cv):.4f} +/- {np.std(recall_cv):.4f}")
print(f"  Mean F1:     {np.mean(f1_cv):.4f} +/- {np.std(f1_cv):.4f}")
print(f"  Mean F2:     {np.mean(f2_cv):.4f} +/- {np.std(f2_cv):.4f}")

# ── Final training ──────────────────────────────────────────────
print("\n--- Final training (full train set) ---")
smote = SMOTE(random_state=42, k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_tr, y_train)

sel = FeatureSelector(n_features=20, method='importance', random_state=42)
X_bal_sel = sel.fit_transform(X_bal, y_bal, prep.get_feature_names())
X_te_sel  = sel.transform(X_te)

iforest_cfg = load_config('configs/models/isolation_forest.yaml')
model = HybridNIDS(rf_params=rf_params, iforest_params=iforest_cfg['hyperparameters'], random_state=42)
model.train(X_bal_sel, y_bal, normal_label='Normal')

class_order = list(model.tier1_model.model.classes_)
preds, tiers = model.predict(X_te_sel)
detailed = model.predict_with_scores(X_te_sel)
all_labels = sorted(set(class_order) | set(y_test) | set(preds))
proba_labels = sorted([l for l in all_labels if l != 'Zero_Day_Anomaly'])

evaluator = NIDSEvaluator(output_dir=str(OUTPUT_DIR))
metrics = evaluator.evaluate(
    y_true=y_test, y_pred=preds,
    y_proba=detailed['tier1_probabilities'],
    labels=all_labels, normal_label='Normal'
)

# ── Overfitting check ───────────────────────────────────────────
train_p, _ = model.predict(X_bal_sel)
train_r = recall_score(y_bal, train_p, average='weighted', zero_division=0)
test_r  = recall_score(y_test, preds, average='weighted', zero_division=0)
gap = train_r - test_r
print(f"\nOverfit check: Train={train_r:.4f} Test={test_r:.4f} Gap={gap:+.4f}")

# ── Latency ─────────────────────────────────────────────────────
sample = X_te_sel[:1]
times = [(time.perf_counter(), model.predict(sample), time.perf_counter()) for _ in range(200)]
latencies = [(t[2]-t[0])*1000 for t in times]
avg_ms = np.mean(latencies)
print(f"Latency: avg={avg_ms:.1f}ms p99={np.percentile(latencies, 99):.1f}ms")

# ── Save report ─────────────────────────────────────────────────
report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'UNSW-NB15',
    'config': 'hardened (200 trees)',
    'cross_validation': {
        'recall': {'mean': float(np.mean(recall_cv)), 'std': float(np.std(recall_cv))},
        'f1': {'mean': float(np.mean(f1_cv)), 'std': float(np.std(f1_cv))},
        'f2': {'mean': float(np.mean(f2_cv)), 'std': float(np.std(f2_cv))},
    },
    'test_metrics': {k: v for k, v in metrics.items()
                     if not isinstance(v, (list, np.ndarray)) and k != 'classification_report'},
    'classification_report': metrics.get('classification_report', ''),
    'overfitting': {'train_recall': float(train_r), 'test_recall': float(test_r), 'gap': float(gap)},
    'latency_avg_ms': float(avg_ms),
}
with open(OUTPUT_DIR / 'unsw_validation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n[Saved] {OUTPUT_DIR / 'unsw_validation_report.json'}")
