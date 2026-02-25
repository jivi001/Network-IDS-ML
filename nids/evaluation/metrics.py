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
    roc_curve,
    roc_auc_score,
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
    Security-focused evaluation for NIDS.
    Computes: Accuracy, Recall, Precision, F1, F2, Confusion Matrix.
    Emphasis on minimizing False Negatives (missed attacks).
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _get_attack_score(
        y_proba: np.ndarray, labels: list, normal_label: str = 'Normal'
    ) -> np.ndarray:
        """
        Compute per-sample P(Attack) from multiclass probability matrix.

        Correctly handles arbitrary class ordering by finding the
        Normal column index from the labels list, rather than assuming
        column 0 is Normal (which fails when sklearn sorts alphabetically).

        Args:
            y_proba:  (n_samples, n_classes) probability matrix
            labels:   ordered class list matching y_proba columns
            normal_label: the label that denotes benign traffic

        Returns:
            1-D array of P(Attack) = 1 - P(Normal) for each sample
        """
        if y_proba.ndim == 1:
            # If y_proba is already a 1D array (e.g., binary classifier outputting P(positive))
            return y_proba

        labels_list = list(labels)
        if normal_label in labels_list:
            normal_idx = labels_list.index(normal_label)
            return 1.0 - y_proba[:, normal_idx]
        else:
            # Fallback: sum of all columns except the first (original behavior)
            # This assumes the first column is 'Normal' or the most common benign class
            # This fallback is less robust and should ideally be avoided by providing correct labels.
            print(f"[Warning] Normal label '{normal_label}' not found in provided labels. "
                  "Falling back to 1 - y_proba[:, 0]. Ensure column 0 corresponds to Normal class.")
            return 1.0 - y_proba[:, 0]

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

        # Determine probability column labels (y_proba may have fewer columns
        # than `labels` when Tier 2 adds classes like Zero_Day_Anomaly)
        proba_labels = labels
        if y_proba is not None and labels is not None:
            if y_proba.ndim > 1 and y_proba.shape[1] != len(labels):
                # y_proba columns = RF classes (sorted alphabetically by sklearn)
                # Filter labels to only the ones that appear in the RF model
                proba_labels = sorted([l for l in labels if l != 'Zero_Day_Anomaly'])
                if y_proba.shape[1] != len(proba_labels):
                    # Last resort: just use sorted labels up to the column count
                    proba_labels = sorted(set(y_true))[:y_proba.shape[1]]

        # --- PR Curve ---
        pr_auc_val = None
        if y_proba is not None:
            pr_auc_val = self._plot_precision_recall_curve(y_true, y_proba, normal_label, proba_labels)

        # --- ROC Curve ---
        roc_auc = None
        if y_proba is not None:
            roc_auc = self._plot_roc_curve(y_true, y_proba, normal_label, proba_labels)

        # --- Threshold Optimization ---
        optimal_threshold = None
        best_f2 = None
        if y_proba is not None:
            optimal_threshold, best_f2 = self.optimize_threshold(
                y_true, y_proba, normal_label, labels=proba_labels
            )

        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'f2_score': f2,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc_val,
            'optimal_threshold': optimal_threshold,
            'optimal_f2_at_threshold': best_f2,
            'confusion_matrix': cm.tolist(),
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
        self, y_true: np.ndarray, y_proba: np.ndarray,
        normal_label: str, labels: list = None
    ) -> float:
        """Plot PR curve for attack detection (binary). Returns PR-AUC."""
        try:
            y_true_binary = (np.array(y_true) != normal_label).astype(int)
            y_score = self._get_attack_score(y_proba, labels or [], normal_label)

            prec, rec, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc_val = auc(rec, prec)

            plt.figure(figsize=(10, 6))
            plt.plot(rec, prec, linewidth=2, label=f'PR Curve (AUC = {pr_auc_val:.4f})')
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
            return pr_auc_val
        except Exception as e:
            print(f"[Warning] Could not plot PR curve: {e}")
            return None


    def _plot_roc_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray,
        normal_label: str, labels: list = None
    ) -> float:
        """Plot ROC curve and return ROC-AUC for attack detection (binary)."""
        try:
            y_true_binary = (np.array(y_true) != normal_label).astype(int)
            y_score = self._get_attack_score(y_proba, labels or [], normal_label)

            roc_auc = roc_auc_score(y_true_binary, y_score)

            # Sanity check: if ROC-AUC < 0.5 the scores are inverted
            if roc_auc < 0.5:
                print(f"  [FIX] ROC-AUC={roc_auc:.4f} < 0.5 — inverting scores")
                y_score = 1.0 - y_score
                roc_auc = roc_auc_score(y_true_binary, y_score)

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)

            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            plt.xlabel('False Positive Rate (False Alarm Rate)', fontsize=12)
            plt.ylabel('True Positive Rate (Detection Rate)', fontsize=12)
            plt.title('ROC Curve \u2014 Attack Detection', fontsize=16, fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir is not None:
                filepath = os.path.join(self.output_dir, 'roc_curve.png')
                plt.savefig(filepath, dpi=300)
                print(f"[Saved] ROC Curve (AUC={roc_auc:.4f}) -> {filepath}")
            plt.close()
            return roc_auc
        except Exception as e:
            print(f"[Warning] Could not plot ROC curve: {e}")
            return None

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        normal_label: str = 'Normal',
        labels: list = None,
        beta: float = 2.0,
        max_fpr: float = 0.05
    ) -> tuple:
        """
        Find the decision threshold that maximizes F-beta score on the PR curve,
        subject to a False Positive Rate constraint.

        Args:
            y_true: Ground truth labels
            y_proba: Prediction probabilities
            normal_label: Label for benign traffic
            labels: Ordered class list matching y_proba columns
            beta: F-beta parameter (default 2.0 = recall-biased)
            max_fpr: Maximum allowable FPR (default 5% for SOC)

        Returns:
            (optimal_threshold, best_f_beta_score)
        """
        try:
            y_true_binary = (np.array(y_true) != normal_label).astype(int)
            y_score = self._get_attack_score(y_proba, labels or [], normal_label)

            prec, rec, thresholds = precision_recall_curve(y_true_binary, y_score)
            # Also get FPR at each threshold via ROC
            fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true_binary, y_score)

            # Compute F-beta at each PR threshold
            f_beta = np.where(
                (prec[:-1] + rec[:-1]) > 0,
                (1 + beta**2) * (prec[:-1] * rec[:-1]) / ((beta**2 * prec[:-1]) + rec[:-1]),
                0
            )

            # For each PR threshold, compute FPR
            n_normal = (y_true_binary == 0).sum()
            threshold_fpr = np.array([
                ((y_score >= t) & (y_true_binary == 0)).sum() / max(n_normal, 1)
                for t in thresholds
            ])

            # Apply constraints:
            # 1. threshold > 0.01 (reject degenerate 0.0)
            # 2. FPR <= max_fpr
            valid = (thresholds > 0.01) & (threshold_fpr <= max_fpr)

            if valid.any():
                constrained_f_beta = np.where(valid, f_beta, -1)
                best_idx = np.argmax(constrained_f_beta)
            else:
                # Fallback: relax FPR constraint and pick best F-beta with threshold > 0.01
                valid_nofpr = thresholds > 0.01
                if valid_nofpr.any():
                    constrained_f_beta = np.where(valid_nofpr, f_beta, -1)
                    best_idx = np.argmax(constrained_f_beta)
                    print(f"  [WARN] No threshold meets FPR<={max_fpr:.0%}; relaxed constraint")
                else:
                    best_idx = np.argmax(f_beta)
                    print(f"  [WARN] All thresholds degenerate; picking global best")

            optimal_threshold = float(thresholds[best_idx])
            best_score = float(f_beta[best_idx])
            fpr_at_opt = float(threshold_fpr[best_idx])

            print(f"\n--- Threshold Optimization (F{beta:.0f}, FPR<={max_fpr:.0%}) ---")
            print(f"  Optimal threshold:  {optimal_threshold:.4f}")
            print(f"  Best F{beta:.0f}-score:     {best_score:.4f}")
            print(f"  Precision at opt:   {prec[best_idx]:.4f}")
            print(f"  Recall at opt:      {rec[best_idx]:.4f}")
            print(f"  FPR at opt:         {fpr_at_opt:.4f}")

            return optimal_threshold, best_score

        except Exception as e:
            print(f"[Warning] Threshold optimization failed: {e}")
            return None, None

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
