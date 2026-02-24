from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from src.feature_selection.selector import FeatureSelectionConfig, build_selector
from src.models.factory import build_model
from src.preprocessing.pipeline import PreprocessingConfig, build_preprocessor


@dataclass
class TrainingPipelineConfig:
    model_name: str
    model_params: dict[str, Any]
    random_state: int
    use_smote: bool
    smote_params: dict[str, Any]
    preprocessing: PreprocessingConfig
    feature_selection: FeatureSelectionConfig


def build_training_pipeline(config: TrainingPipelineConfig) -> Pipeline:
    """Build leakage-safe pipeline: preprocessing -> optional SMOTE -> feature selection -> model."""
    steps: list[tuple[str, Any]] = [
        ("preprocessor", build_preprocessor(config.preprocessing))
    ]
    if config.use_smote:
        steps.append(
            ("smote", SMOTE(random_state=config.random_state, **config.smote_params))
        )
    steps.append(("selector", build_selector(config.feature_selection)))
    steps.append(
        (
            "model",
            build_model(config.model_name, config.model_params, config.random_state),
        )
    )
    return Pipeline(steps=steps)
