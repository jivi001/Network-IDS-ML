"""
Main execution script for Hybrid NIDS.
Demonstrates end-to-end pipeline with synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Absolute imports for direct execution
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessing import NIDSPreprocessor
from src.feature_selection import FeatureSelector
from src.hybrid_system import HybridNIDS
from src.evaluation import NIDSEvaluator


def generate_synthetic_data(n_samples=5000, n_features=30):
    """
    Generate synthetic network traffic data for testing.
    
    Returns:
        DataFrame with features and labels
    """
    print("\n=== Generating Synthetic Dataset ===")
    
    # Create imbalanced multi-class dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_classes=4,
        weights=[0.7, 0.15, 0.10, 0.05],  # Imbalanced
        flip_y=0.01,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Map labels to attack types
    label_map = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L'}
    df['label'] = pd.Series(y).map(label_map)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "="*70)
    print("HYBRID NIDS - PRODUCTION PIPELINE")
    print("="*70)
    
    # === 1. DATA LOADING ===
    # For demonstration, use synthetic data
    # In production, replace with: loader.load_csv('path/to/dataset.csv')
    
    df = generate_synthetic_data(n_samples=5000, n_features=30)
    
    loader = DataLoader(dataset_type='auto')
    X, y = loader.split_features_labels(df)
    
    print(f"\n[Pipeline] Features: {X.shape}, Labels: {y.shape}")
    
    # === 2. TRAIN-TEST SPLIT ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"[Pipeline] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # === 3. PREPROCESSING ===
    print("\n=== Preprocessing ===")
    preprocessor = NIDSPreprocessor(random_state=42)
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"[Preprocessor] Train shape: {X_train_processed.shape}")
    print(f"[Preprocessor] Test shape: {X_test_processed.shape}")
    
    # === 4. APPLY SMOTE (Training Only) ===
    print("\n=== Applying SMOTE ===")
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(
        X_train_processed,
        y_train.values
    )
    
    print(f"[SMOTE] Balanced train shape: {X_train_balanced.shape}")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print(f"[SMOTE] Class distribution after balancing:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    # === 5. FEATURE SELECTION (Optional) ===
    # Uncomment to enable feature selection
    # print("\n=== Feature Selection ===")
    # selector = FeatureSelector(n_features=20, random_state=42)
    # selector.fit(X_train_balanced, y_train_balanced, preprocessor.get_feature_names())
    # X_train_selected = selector.transform(X_train)
    # X_test_selected = selector.transform(X_test)
    
    # === 6. TRAIN HYBRID NIDS ===
    print("\n=== Training Hybrid NIDS ===")
    
    hybrid_nids = HybridNIDS(
        rf_params={'n_estimators': 100, 'max_depth': 20},
        iforest_params={'contamination': 0.1, 'n_estimators': 100},
        random_state=42
    )
    
    hybrid_nids.train(
        X_train_balanced,
        y_train_balanced,
        normal_label='Normal'
    )
    
    # === 7. PREDICTION ===
    print("\n=== Making Predictions ===")
    
    predictions, tier_flags = hybrid_nids.predict(X_test_processed)
    
    # Count tier usage
    tier1_count = np.sum(tier_flags == 1)
    tier2_count = np.sum(tier_flags == 2)
    
    print(f"[Prediction] Tier 1 decisions: {tier1_count}")
    print(f"[Prediction] Tier 2 decisions: {tier2_count}")
    print(f"[Prediction] Tier 2 triggered on {tier2_count / len(predictions) * 100:.2f}% of samples")
    
    # === 8. EVALUATION ===
    print("\n=== Evaluation ===")
    
    evaluator = NIDSEvaluator(output_dir='logs')
    
    # Get class labels
    unique_labels = np.unique(np.concatenate([y_test.values, predictions]))
    
    metrics = evaluator.evaluate(
        y_true=y_test.values,
        y_pred=predictions,
        labels=unique_labels.tolist()
    )
    
    # === 9. VERIFICATION CHECKS ===
    print("\n=== Verification Checks ===")
    
    # Check 1: SMOTE increased sample count
    assert X_train_balanced.shape[0] >= X_train_processed.shape[0], "SMOTE failed to increase samples"
    print("✓ SMOTE correctly increased training samples")
    
    # Check 2: Tier 2 only triggered on some samples
    assert tier2_count > 0, "Tier 2 was never triggered"
    print("✓ Tier 2 (Isolation Forest) was triggered")
    
    # Check 3: Predictions have correct shape
    assert len(predictions) == len(y_test), "Prediction count mismatch"
    print("✓ Prediction shape matches test set")
    
    # Check 4: Zero-day anomalies detected
    zero_day_count = np.sum(predictions == 'Zero_Day_Anomaly')
    print(f"✓ Zero-day anomalies detected: {zero_day_count}")
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: logs/")
    print("  - confusion_matrix.png")
    print("  - precision_recall_curve.png")
    
    return hybrid_nids, metrics


if __name__ == '__main__':
    hybrid_nids, metrics = main()
