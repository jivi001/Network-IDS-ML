from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Tree-based SHAP explainability utility."""

    def __init__(self, trained_pipeline: Any):
        self.pipeline = trained_pipeline

    def explain_top_features(
        self, request_df: pd.DataFrame, top_k: int = 5
    ) -> list[dict[str, float]]:
        model = self.pipeline.named_steps["model"]
        transformed = self.pipeline.named_steps["selector"].transform(
            self.pipeline.named_steps["preprocessor"].transform(request_df)
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        raw = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        feature_scores = np.abs(raw)
        ranked_indices = np.argsort(feature_scores)[::-1][:top_k]
        return [
            {"feature_index": float(index), "impact": float(feature_scores[index])}
            for index in ranked_indices
        ]
