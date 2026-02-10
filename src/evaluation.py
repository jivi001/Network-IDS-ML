"""
Evaluation metrics and visualization for NIDS.
Focus on Recall, Precision-Recall curves, and security-oriented metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    recall_score,
    precision_score
)
from typing import Optional
import os


class NIDSEvaluator:
    """
    Evaluation suite for intrusion detection systems.
    Prioritizes Recall (detection rate) over Precision.
    """
    
    def __init__(self, output_dir: str = 'logs'):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[list] = None
    ) -> dict:
        """
        Comprehensive evaluation.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for PR curves)
            labels: List of label names
            
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*60)
        print("NIDS EVALUATION REPORT")
        print("="*60)
        
        # Classification report
        print("\n--- Classification Report ---")
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, labels)
        
        # Overall metrics
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n--- Overall Metrics ---")
        print(f"Weighted Recall (Detection Rate): {recall:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Security-specific metrics
        self._compute_security_metrics(y_true, y_pred)
        
        # Precision-Recall curve (if probabilities provided)
        if y_proba is not None:
            self._plot_precision_recall_curve(y_true, y_proba)
        
        return {
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def _compute_security_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute security-focused metrics.
        
        In NIDS context:
        - False Negative (FN): Attack missed (CRITICAL)
        - False Positive (FP): Normal flagged as attack (Alert fatigue)
        """
        # Binary classification: Attack vs Normal
        # Convert to binary (assuming 'Normal' or similar is the negative class)
        unique_labels = np.unique(y_true)
        
        # Heuristic: assume first label alphabetically is "Normal"
        normal_label = sorted(unique_labels)[0]
        
        y_true_binary = (y_true != normal_label).astype(int)
        y_pred_binary = (y_pred != normal_label).astype(int)
        
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        
        print(f"\n--- Security Metrics (Binary: Attack Detection) ---")
        print(f"True Positives (Attacks Detected): {tp}")
        print(f"False Negatives (Attacks Missed): {fn} ⚠️")
        print(f"False Positives (False Alarms): {fp}")
        print(f"True Negatives (Normal Correctly Identified): {tn}")
        
        if (tp + fn) > 0:
            attack_detection_rate = tp / (tp + fn)
            print(f"Attack Detection Rate (Recall): {attack_detection_rate:.4f}")
        
        if (fp + tn) > 0:
            false_alarm_rate = fp / (fp + tn)
            print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    
    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[list] = None
    ):
        """
        Plot and save confusion matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels or range(len(cm)),
            yticklabels=labels or range(len(cm))
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300)
        print(f"\n[Saved] Confusion Matrix → {filepath}")
        plt.close()
    
    def _plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ):
        """
        Plot Precision-Recall curve.
        
        For multi-class, we plot for "attack" class (binary: attack vs normal).
        """
        # Convert to binary
        unique_labels = np.unique(y_true)
        normal_label = sorted(unique_labels)[0]
        
        y_true_binary = (y_true != normal_label).astype(int)
        
        # If y_proba is multi-class, take max probability of attack classes
        if y_proba.ndim > 1:
            # Assume first column is "normal", rest are attacks
            y_score = 1 - y_proba[:, 0]  # Probability of being an attack
        else:
            y_score = y_proba
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall (Attack Detection Rate)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
        plt.savefig(filepath, dpi=300)
        print(f"[Saved] Precision-Recall Curve → {filepath}")
        plt.close()
    
    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list,
        top_n: int = 20
    ):
        """
        Plot feature importance from Random Forest.
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(top_n),
            importances[indices],
            align='center'
        )
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300)
        print(f"[Saved] Feature Importance → {filepath}")
        plt.close()
