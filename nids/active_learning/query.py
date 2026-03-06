"""
Uncertainty + Diversity Active Learning Query Strategy for NIDS.

SOC analysts have limited labeling bandwidth. This module selects the
most informative unlabeled samples for analyst review using:

  1. Uncertainty sampling — entropy of Tier 1 class probabilities
  2. Diversity clustering  — K-Means over the uncertain subset so the
                             analyst sees varied samples, not duplicates

The combined strategy is known as Informative Diversity Sampling.

Usage:
    query = UncertaintyDiversityQuery(budget=50)
    indices = query.select(tier1_model, X_unlabeled)
    # Present X_unlabeled[indices] to SOC analyst for labeling
"""

import numpy as np
from typing import Optional


class UncertaintyDiversityQuery:
    """
    Active learning query strategy combining uncertainty + diversity.

    Args:
        budget (int): Max samples to return per query round.
        uncertainty_pool (int): Size of the uncertainty-filtered pool
                                 before diversity clustering.
                                 Must satisfy budget <= uncertainty_pool.
        random_state (int): Random seed for K-Means.
    """

    def __init__(
        self,
        budget: int = 50,
        uncertainty_pool: int = 500,
        random_state: int = 42,
    ):
        self.budget = budget
        self.uncertainty_pool = max(uncertainty_pool, budget)
        self.random_state = random_state

    def select(self, tier1_model, X_unlabeled: np.ndarray) -> np.ndarray:
        """
        Select the most informative sample indices from X_unlabeled.

        Args:
            tier1_model: Fitted model with predict_proba() method.
                         Can be SupervisedModel or StackingEnsemble.
            X_unlabeled: Feature matrix of unlabeled samples.

        Returns:
            1-D array of selected indices into X_unlabeled (length <= budget).
        """
        n = len(X_unlabeled)
        if n == 0:
            return np.array([], dtype=int)

        budget = min(self.budget, n)
        pool_size = min(self.uncertainty_pool, n)

        # --- Step 1: Compute uncertainty (Shannon entropy) ---
        proba = tier1_model.predict_proba(X_unlabeled)
        # Clip to avoid log(0)
        proba = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(proba * np.log2(proba), axis=1)   # (n,)

        # Select top uncertainty_pool-size samples
        top_uncertain_idx = np.argsort(entropy)[::-1][:pool_size]
        X_pool = X_unlabeled[top_uncertain_idx]

        # --- Step 2: Diversity via K-Means clustering ---
        if budget >= pool_size:
            # All pool samples fit within budget — return all
            return top_uncertain_idx[:budget]

        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=budget,
            random_state=self.random_state,
            n_init=3,
        ).fit(X_pool)

        # For each cluster, pick the sample with highest entropy
        selected = []
        pool_entropy = entropy[top_uncertain_idx]
        for cluster_id in range(budget):
            member_mask = km.labels_ == cluster_id
            if not member_mask.any():
                continue
            member_entropies = pool_entropy[member_mask]
            member_pool_indices = np.where(member_mask)[0]
            best_in_cluster = member_pool_indices[member_entropies.argmax()]
            selected.append(top_uncertain_idx[best_in_cluster])

        return np.array(selected, dtype=int)

    def entropy_scores(self, tier1_model, X: np.ndarray) -> np.ndarray:
        """Return per-sample Shannon entropy of class probabilities."""
        proba = np.clip(tier1_model.predict_proba(X), 1e-10, 1.0)
        return -np.sum(proba * np.log2(proba), axis=1)
