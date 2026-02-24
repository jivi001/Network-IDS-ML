from src.feature_selection.selector import FeatureSelectionConfig
from src.preprocessing.pipeline import PreprocessingConfig
from src.training.pipeline_builder import (
    TrainingPipelineConfig,
    build_training_pipeline,
)


def test_build_training_pipeline_contains_steps() -> None:
    config = TrainingPipelineConfig(
        model_name="random_forest",
        model_params={"n_estimators": 10},
        random_state=42,
        use_smote=True,
        smote_params={},
        preprocessing=PreprocessingConfig(
            numeric_features=["x"], categorical_features=["c"]
        ),
        feature_selection=FeatureSelectionConfig(k_best=1),
    )
    pipeline = build_training_pipeline(config)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["preprocessor", "smote", "selector", "model"]
