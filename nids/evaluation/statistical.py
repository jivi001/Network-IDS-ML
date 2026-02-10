"""
Statistical significance testing for model comparison.
Implements repeated k-fold cross-validation and paired t-tests.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone


class StatisticalEvaluator:
    """
    Perform statistical significance testing for model comparison.
    Ensures model improvements are not due to random chance.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical evaluator.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def compare_models(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        test: str = 'paired_t'
    ) -> Dict[str, any]:
        """
        Compare two models using statistical test.
        
        Args:
            model_a_scores: Performance scores from model A (e.g., recall from k-fold CV)
            model_b_scores: Performance scores from model B
            test: Statistical test to use ('paired_t' or 'wilcoxon')
            
        Returns:
            Dictionary with test results
        """
        if len(model_a_scores) != len(model_b_scores):
            raise ValueError("Score lists must have same length")
        
        model_a_scores = np.array(model_a_scores)
        model_b_scores = np.array(model_b_scores)
        
        # Compute statistics
        mean_a = np.mean(model_a_scores)
        std_a = np.std(model_a_scores, ddof=1)
        mean_b = np.mean(model_b_scores)
        std_b = np.std(model_b_scores, ddof=1)
        
        # Perform test
        if test == 'paired_t':
            statistic, p_value = ttest_rel(model_a_scores, model_b_scores)
        elif test == 'wilcoxon':
            statistic, p_value = wilcoxon(model_a_scores, model_b_scores)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        # Determine which model is better
        if is_significant:
            better_model = 'A' if mean_a > mean_b else 'B'
        else:
            better_model = 'No significant difference'
        
        return {
            'model_a_mean': float(mean_a),
            'model_a_std': float(std_a),
            'model_b_mean': float(mean_b),
            'model_b_std': float(std_b),
            'test_statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'is_significant': is_significant,
            'better_model': better_model,
            'test_used': test
        }
    
    def cross_validation_with_stats(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 3,
        metric_func: callable = None
    ) -> Dict[str, any]:
        """
        Perform repeated stratified k-fold cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Labels
            n_splits: Number of folds
            n_repeats: Number of repetitions
            metric_func: Function to compute metric (default: accuracy)
            
        Returns:
            Dictionary with CV results and statistics
        """
        if metric_func is None:
            from sklearn.metrics import accuracy_score
            metric_func = accuracy_score
        
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42
        )
        
        scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            score = metric_func(y_test, y_pred)
            scores.append(score)
        
        scores = np.array(scores)
        
        return {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores, ddof=1)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'total_runs': len(scores)
        }
    
    def compute_confidence_interval(
        self,
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using bootstrap.
        
        Args:
            scores: List of performance scores
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        scores = np.array(scores)
        alpha = 1 - confidence
        
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(scores, lower_percentile)
        upper = np.percentile(scores, upper_percentile)
        
        return float(lower), float(upper)
