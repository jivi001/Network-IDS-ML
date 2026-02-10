"""
Security-focused evaluation suite for NIDS.
Prioritizes Recall (detection rate) and computes F2-Score.
Generates confusion matrix, PR curves, and feature importance plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    accuracy_score
)
from typing import Optional, List
from tabulate import tabulate
import os


class NIDSEvaluator:
    """
    Evaluation suite for intrusion detection systems.
    Prioritizes Recall (minimize missed attacks / False Negatives).
    """

    def __init__(self, output_dir: Optional[str] = 'logs'):
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[list] = None,
        normal_label: str = 'Normal'
    ) -> dict:
        """
        Comprehensive security-focused evaluation.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for PR curves)
            labels: List of label names
            normal_label: Label string for benign traffic

        Returns:
            Dictionary of all computed metrics
        """
        print("\n" + "=" * 70)
        print("  NIDS EVALUATION REPORT")
        print("=" * 70)

        # --- Classification Report ---
        print("\n--- Per-Class Classification Report ---")
        report = classification_report(
            y_true, y_pred, target_names=labels,
            zero_division=0, output_dict=False
        )
        print(report)

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self._plot_confusion_matrix(cm, labels)

        # --- Overall Weighted Metrics ---
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        metrics_table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Weighted Recall (Detection Rate)", f"{recall:.4f}"],
            ["Weighted Precision", f"{precision:.4f}"],
            ["Weighted F1-Score", f"{f1:.4f}"],
            ["Weighted F2-Score (Recall-biased)", f"{f2:.4f}"],
        ]
        print("\n--- Overall Metrics ---")
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

        # --- Security Metrics (Binary: Attack vs Normal) ---
        sec_metrics = self._compute_security_metrics(y_true, y_pred, normal_label)

        # --- PR Curve ---
        if y_proba is not None:
            self._plot_precision_recall_curve(y_true, y_proba, normal_label)

        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'f2_score': f2,
            'confusion_matrix': cm,
            'classification_report': report,
            **sec_metrics
        }

    def _compute_security_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, normal_label: str
    ) -> dict:
        """
        Binary security metrics: Attack vs Normal.
        FN (missed attacks) is the most critical metric.
        """
        y_true_binary = (np.array(y_true) != normal_label).astype(int)
        y_pred_binary = (np.array(y_pred) != normal_label).astype(int)

        tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
        fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
        fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
        tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))

        attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        sec_table = [
            ["True Positives (Attacks Detected)", tp],
            ["False Negatives (Attacks MISSED) [CRITICAL]", fn],
            ["False Positives (False Alarms)", fp],
            ["True Negatives (Normal OK)", tn],
            ["Attack Detection Rate (Recall)", f"{attack_recall:.4f}"],
            ["False Alarm Rate", f"{false_alarm_rate:.4f}"],
        ]
        print("\n--- Security Metrics (Binary: Attack Detection) ---")
        print(tabulate(sec_table, headers=["Metric", "Value"], tablefmt="grid"))

        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'attack_detection_rate': attack_recall,
            'false_alarm_rate': false_alarm_rate,
        }

    def _plot_confusion_matrix(
        self, cm: np.ndarray, labels: Optional[list] = None
    ):
        """Plot and save confusion matrix heatmap."""
        plt.figure(figsize=(max(8, len(cm)), max(6, len(cm) * 0.8)))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels or range(len(cm)),
            yticklabels=labels or range(len(cm))
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if self.output_dir is not None:
            filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300)
            print(f"\n[Saved] Confusion Matrix -> {filepath}")
        plt.close()

    def _plot_precision_recall_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray, normal_label: str
    ):
        """Plot PR curve for attack detection (binary)."""
        try:
            y_true_binary = (np.array(y_true) != normal_label).astype(int)

            if y_proba.ndim > 1:
                # Sum probabilities of all attack classes
                y_score = 1 - y_proba[:, 0]
            else:
                y_score = y_proba

            prec, rec, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc = auc(rec, prec)

            plt.figure(figsize=(10, 6))
            plt.plot(rec, prec, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
            plt.xlabel('Recall (Attack Detection Rate)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir is not None:
                filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
                plt.savefig(filepath, dpi=300)
                print(f"[Saved] Precision-Recall Curve -> {filepath}")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not plot PR curve: {e}")

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list,
        top_n: int = 20
    ):
        """Plot top-N feature importances from Random Forest."""
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(
            range(top_n),
            importances[indices],
            align='center',
            color='steelblue'
        )
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance (Gini Decrease)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if self.output_dir is not None:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=300)
            print(f"[Saved] Feature Importance -> {filepath}")
        plt.close()

    def evaluate_tier_stats(
        self, tier_flags: np.ndarray, predictions: np.ndarray
    ):
        """Print tier usage statistics."""
        tier1_count = int(np.sum(tier_flags == 1))
        tier2_count = int(np.sum(tier_flags == 2))
        zd_count = int(np.sum(predictions == 'Zero_Day_Anomaly'))
        total = len(tier_flags)

        tier_table = [
            ["Tier 1 (RF) Decisions", tier1_count, f"{tier1_count/total*100:.1f}%"],
            ["Tier 2 (iForest) Decisions", tier2_count, f"{tier2_count/total*100:.1f}%"],
            ["Zero-Day Anomalies Flagged", zd_count, f"{zd_count/total*100:.1f}%"],
        ]
        print("\n--- Tier Usage Statistics ---")
        print(tabulate(tier_table, headers=["Tier", "Count", "Percentage"], tablefmt="grid"))
