from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_selection import SelectKBest, f_classif


@dataclass
class FeatureSelectionConfig:
    k_best: int


def build_selector(config: FeatureSelectionConfig) -> SelectKBest:
    """Create a deterministic feature selector."""
    return SelectKBest(score_func=f_classif, k=config.k_best)
