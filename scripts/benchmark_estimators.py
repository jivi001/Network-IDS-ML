"""
Quick benchmark: n_estimators = 300 vs 200 vs 150
Measures ROC-AUC, PR-AUC, Recall, and single-prediction latency.
"""
import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.data import DataLoader
from nids.preprocessing import NIDSPreprocessor
from nids.features.selection import FeatureSelector
from nids.models import HybridNIDS
from nids.evaluation import NIDSEvaluator
from nids.utils.config import load_config
from imblearn.over_sampling import SMOTE

# Load & preprocess
loader = DataLoader(dataset_type='nsl_kdd')
df_train = loader.load_csv('data/raw/nsl_kdd_train.csv')
df_test  = loader.load_csv('data/raw/nsl_kdd_test.csv')
X_train, y_train = loader.split_features_labels(df_train)
X_test, y_test   = loader.split_features_labels(df_test)

prep = NIDSPreprocessor(random_state=42)
X_tr = prep.fit_transform(X_train)
X_te = prep.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_tr, y_train)

sel = FeatureSelector(n_features=20, method='importance', random_state=42)
X_bal_sel = sel.fit_transform(X_bal, y_bal, prep.get_feature_names())
X_te_sel  = sel.transform(X_te)

iforest_cfg = load_config('configs/models/isolation_forest.yaml')
base_rf = load_config('configs/models/random_forest_hardened.yaml')['hyperparameters']

results = []

for n_est in [300, 200, 150]:
    print(f"\n{'='*50}")
    print(f"  n_estimators = {n_est}")
    print(f"{'='*50}")

    rf_params = {**base_rf, 'n_estimators': n_est}
    model = HybridNIDS(rf_params=rf_params, iforest_params=iforest_cfg['hyperparameters'], random_state=42)
    model.train(X_bal_sel, y_bal, normal_label='Normal')

    class_order = list(model.tier1_model.model.classes_)
    preds, _ = model.predict(X_te_sel)
    detailed = model.predict_with_scores(X_te_sel)

    all_labels = sorted(set(class_order) | set(y_test) | set(preds))
    proba_labels = sorted([l for l in all_labels if l != 'Zero_Day_Anomaly'])

    evaluator = NIDSEvaluator(output_dir=None)
    y_proba = detailed['tier1_probabilities']
    y_true_bin = (np.array(y_test) != 'Normal').astype(int)
    y_score = evaluator._get_attack_score(y_proba, proba_labels, 'Normal')

    from sklearn.metrics import roc_auc_score, recall_score, f1_score, fbeta_score
    from sklearn.metrics import precision_recall_curve, auc as sk_auc

    roc = roc_auc_score(y_true_bin, y_score)
    prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
    pr = sk_auc(rec, prec)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, average='weighted', zero_division=0)

    # Latency
    sample = X_te_sel[:1]
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(times)
    p99_ms = np.percentile(times, 99)

    row = {
        'n_estimators': n_est, 'roc_auc': round(roc, 4), 'pr_auc': round(pr, 4),
        'recall': round(recall, 4), 'f2': round(f2, 4),
        'latency_avg_ms': round(avg_ms, 1), 'latency_p99_ms': round(p99_ms, 1)
    }
    results.append(row)
    print(f"  ROC-AUC:  {roc:.4f}")
    print(f"  PR-AUC:   {pr:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  F2:       {f2:.4f}")
    print(f"  Latency:  avg={avg_ms:.1f}ms  p99={p99_ms:.1f}ms")

print("\n\n" + "="*60)
print("  COMPARISON TABLE")
print("="*60)
print(f"  {'Trees':>6} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Recall':>8} | {'F2':>8} | {'Avg ms':>8} | {'P99 ms':>8}")
print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for r in results:
    print(f"  {r['n_estimators']:>6} | {r['roc_auc']:>8} | {r['pr_auc']:>8} | {r['recall']:>8} | {r['f2']:>8} | {r['latency_avg_ms']:>8} | {r['latency_p99_ms']:>8}")

# Save
with open('experiments/validation_report/estimator_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] estimator_benchmark.json")
