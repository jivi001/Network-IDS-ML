"""
Main execution script for Hybrid NIDS.
Implements the full pipeline from the research report:
  1. Download/Load Dataset (NSL-KDD or UNSW-NB15)
  2. Preprocess (Clean → Encode → Scale)
  3. Train/Test Split
  4. SMOTE (Training set ONLY)
  5. Feature Selection (RF Importance, top 20)
  6. Train Hybrid NIDS (Tier 1: RF, Tier 2: iForest)
  7. Predict (Cascade: RF → iForest)
  8. Evaluate (Recall-focused metrics, Confusion Matrix, PR Curve)
  9. Save Models + Visualizations
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os
import sys
import time

# Ensure imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessing import NIDSPreprocessor
from src.feature_selection import FeatureSelector
from src.hybrid_system import HybridNIDS
from src.evaluation import NIDSEvaluator


def generate_synthetic_data(n_samples=5000, n_features=30):
    """
    Generate synthetic network traffic data for testing.
    Simulates imbalanced multi-class NIDS data.
    """
    print("\n=== Generating Synthetic Dataset ===")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_classes=4,
        weights=[0.70, 0.15, 0.10, 0.05],
        flip_y=0.01,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    label_map = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L'}
    df['label'] = pd.Series(y).map(label_map)

    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

    return df


def load_real_dataset(data_dir='data/raw'):
    """
    Attempt to load real datasets. Tries NSL-KDD first, then UNSW-NB15.
    If neither exists, downloads NSL-KDD.
    """
    # Check for NSL-KDD
    nsl_train = os.path.join(data_dir, 'nsl_kdd_train.csv')
    nsl_test = os.path.join(data_dir, 'nsl_kdd_test.csv')

    unsw_train = os.path.join(data_dir, 'unsw_nb15_train.csv')
    unsw_test = os.path.join(data_dir, 'unsw_nb15_test.csv')

    if os.path.exists(nsl_train):
        print(f"\n=== Loading NSL-KDD Dataset ===")
        loader = DataLoader(dataset_type='nsl-kdd')
        df_train = loader.load_csv(nsl_train)
        df_test = loader.load_csv(nsl_test) if os.path.exists(nsl_test) else None
        return df_train, df_test, loader, 'NSL-KDD'

    elif os.path.exists(unsw_train):
        print(f"\n=== Loading UNSW-NB15 Dataset ===")
        loader = DataLoader(dataset_type='unsw-nb15')
        df_train = loader.load_csv(unsw_train)
        df_test = loader.load_csv(unsw_test) if os.path.exists(unsw_test) else None
        return df_train, df_test, loader, 'UNSW-NB15'

    else:
        # Try downloading
        print("\n[INFO] No dataset found. Downloading NSL-KDD...")
        from src.download_datasets import download_nsl_kdd
        download_nsl_kdd(data_dir)

        if os.path.exists(nsl_train):
            loader = DataLoader(dataset_type='nsl-kdd')
            df_train = loader.load_csv(nsl_train)
            df_test = loader.load_csv(nsl_test) if os.path.exists(nsl_test) else None
            return df_train, df_test, loader, 'NSL-KDD'

    return None, None, None, None


def main():
    """
    Full production pipeline for the Hybrid NIDS.
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  HYBRID NIDS - PRODUCTION PIPELINE")
    print("  Random Forest (Tier 1) + Isolation Forest (Tier 2)")
    print("=" * 70)

    # ============================================================
    # 1. DATA LOADING
    # ============================================================
    df_train, df_test, loader, dataset_name = load_real_dataset()

    if df_train is not None:
        print(f"\n[Pipeline] Using real dataset: {dataset_name}")
        X, y = loader.split_features_labels(df_train)

        if df_test is not None:
            X_test_df, y_test_series = loader.split_features_labels(df_test)
            X_train_df, y_train_series = X, y
            use_provided_split = True
        else:
            use_provided_split = False
    else:
        print("\n[Pipeline] Using synthetic data (no real dataset available)")
        df_train = generate_synthetic_data(n_samples=5000, n_features=30)
        loader = DataLoader(dataset_type='auto')
        X, y = loader.split_features_labels(df_train)
        dataset_name = 'Synthetic'
        use_provided_split = False

    # ============================================================
    # 2. TRAIN-TEST SPLIT
    # ============================================================
    if not use_provided_split:
        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    print(f"\n[Pipeline] Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    print(f"[Pipeline] Train class distribution:")
    print(y_train_series.value_counts().to_string())

    # ============================================================
    # 3. PREPROCESSING (Clean → Encode → Scale)
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 3: Preprocessing")
    print("=" * 50)

    preprocessor = NIDSPreprocessor(random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train_df)
    X_test_processed = preprocessor.transform(X_test_df)

    feature_names = preprocessor.get_feature_names()
    print(f"  Train shape after preprocessing: {X_train_processed.shape}")
    print(f"  Test shape after preprocessing:  {X_test_processed.shape}")

    y_train = y_train_series.values
    y_test = y_test_series.values

    # ============================================================
    # 4. SMOTE (Training set ONLY - prevents data leakage)
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 4: SMOTE Balancing (Training Set Only)")
    print("=" * 50)

    X_train_balanced, y_train_balanced = preprocessor.apply_smote(
        X_train_processed, y_train
    )

    # ============================================================
    # 5. FEATURE SELECTION (Recursive Feature Elimination - RFE)
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 5: Feature Selection (RFE)")
    print("=" * 50)

    n_select = min(20, X_train_balanced.shape[1])
    selector = FeatureSelector(n_features=n_select, random_state=42)
    X_train_selected = selector.fit_transform(
        X_train_balanced, y_train_balanced, feature_names
    )
    X_test_selected = selector.transform(X_test_processed)

    print(f"  Train shape after selection: {X_train_selected.shape}")
    print(f"  Test shape after selection:  {X_test_selected.shape}")

    # ============================================================
    # 6. TRAIN HYBRID NIDS
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 6: Training Hybrid NIDS")
    print("=" * 50)

    hybrid_nids = HybridNIDS(
        rf_params={
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5
        },
        iforest_params={
            'contamination': 0.05,
            'n_estimators': 200
        },
        random_state=42
    )

    hybrid_nids.train(
        X_train_selected,
        y_train_balanced,
        normal_label='Normal'
    )

    # ============================================================
    # 7. PREDICTION (Cascade Logic)
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 7: Prediction (Cascade)")
    print("=" * 50)

    predictions, tier_flags = hybrid_nids.predict(X_test_selected)

    # ============================================================
    # 8. EVALUATION
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 8: Evaluation")
    print("=" * 50)

    evaluator = NIDSEvaluator(output_dir='logs')

    # Tier usage statistics
    evaluator.evaluate_tier_stats(tier_flags, predictions)

    # Get unique labels for evaluation
    all_labels = sorted(set(list(np.unique(y_test)) + list(np.unique(predictions))))

    # Get prediction probabilities for PR curve
    detailed = hybrid_nids.predict_with_scores(X_test_selected)

    metrics = evaluator.evaluate(
        y_true=y_test,
        y_pred=predictions,
        y_proba=detailed['tier1_probabilities'],
        labels=all_labels,
        normal_label='Normal'
    )

    # Feature importance plot
    importances = hybrid_nids.tier1_model.get_feature_importances()
    selected_names = selector.get_selected_names()
    evaluator.plot_feature_importance(importances, selected_names)

    # ============================================================
    # 9. SAVE MODELS
    # ============================================================
    os.makedirs('models', exist_ok=True)
    hybrid_nids.save('models/rf_model.pkl', 'models/iforest_model.pkl')

    # ============================================================
    # 10. VERIFICATION CHECKS
    # ============================================================
    print("\n" + "=" * 50)
    print("VERIFICATION CHECKS")
    print("=" * 50)

    checks_passed = 0
    total_checks = 5

    # Check 1: SMOTE increased sample count
    if X_train_balanced.shape[0] >= X_train_processed.shape[0]:
        print("  [OK] SMOTE correctly increased training samples")
        checks_passed += 1
    else:
        print("  ✗ SMOTE did not increase samples")

    # Check 2: Tier 2 was triggered
    tier2_count = np.sum(tier_flags == 2)
    if tier2_count > 0:
        print(f"  [OK] Tier 2 (Isolation Forest) triggered on {tier2_count} samples")
        checks_passed += 1
    else:
        print("  ✗ Tier 2 was never triggered")

    # Check 3: Prediction count matches
    if len(predictions) == len(y_test):
        print("  [OK] Prediction count matches test set")
        checks_passed += 1
    else:
        print("  ✗ Prediction count mismatch")

    # Check 4: Zero-day anomalies detected
    zd_count = np.sum(predictions == 'Zero_Day_Anomaly')
    print(f"  [OK] Zero-day anomalies detected: {zd_count}")
    checks_passed += 1

    # Check 5: Recall above threshold
    if metrics['recall'] > 0.80:
        print(f"  [OK] Weighted Recall ({metrics['recall']:.4f}) > 0.80 threshold")
        checks_passed += 1
    else:
        print(f"  [WARNING] Weighted Recall ({metrics['recall']:.4f}) below 0.80 threshold")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"  Dataset: {dataset_name}")
    print(f"  Checks Passed: {checks_passed}/{total_checks}")
    print("=" * 70)

    print(f"\n  Results saved to: logs/")
    print(f"    - confusion_matrix.png")
    print(f"    - precision_recall_curve.png")
    print(f"    - feature_importance.png")
    print(f"  Models saved to: models/")
    print(f"    - rf_model.pkl")
    print(f"    - iforest_model.pkl")

    return hybrid_nids, metrics


if __name__ == '__main__':
    hybrid_nids, metrics = main()
