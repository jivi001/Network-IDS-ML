from sklearn.metrics import precision_score, recall_score, average_precision_score, classification_report
import numpy as np

def evaluate_soc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, normal_label: str = 'Normal'):
    """Dual-layer SOC evaluation methodology."""
    
    # Layer 1: Binary Detection (Any Attack vs Normal)
    # Both Zero_Day_Anomaly and Suspicious_Low_Conf_Attack count as positive detections
    y_true_bin = (y_true != normal_label).astype(int)
    y_pred_bin = (y_pred != normal_label).astype(int)
    
    # Safe PR-AUC using max probability score if available
    max_proba = np.max(y_proba, axis=1) if y_proba is not None else y_pred_bin
    
    layer1_metrics = {
        'binary_recall': round(recall_score(y_true_bin, y_pred_bin), 4),
        'binary_precision': round(precision_score(y_true_bin, y_pred_bin), 4),
        'pr_auc': round(average_precision_score(y_true_bin, max_proba), 4)
    }

    # Layer 2: Multiclass Attribution (Evaluating only the known attacks)
    # We only care about how well Tier 1 classified the attack IF it was an attack.
    attack_mask = y_true != normal_label
    layer2_metrics = {}
    
    if attack_mask.sum() > 0:
        y_true_attacks = y_true[attack_mask]
        y_pred_attacks = y_pred[attack_mask]
        
        layer2_metrics['attribution_report'] = classification_report(
            y_true_attacks, y_pred_attacks, zero_division=0, output_dict=True
        )

    return {"layer1_detection": layer1_metrics, "layer2_attribution": layer2_metrics}