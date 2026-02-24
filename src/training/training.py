from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.evaluation.metrics import compute_binary_metrics
from src.feature_selection.selector import FeatureSelectionConfig
from src.preprocessing import validate_dataset_schema
from src.preprocessing.pipeline import PreprocessingConfig
from src.training.pipeline_builder import (
    TrainingPipelineConfig,
    build_training_pipeline,
)
from src.training.threshold_optimizer import optimize_f2_threshold
from src.utils.artifacts import log_to_mlflow, persist_training_bundle
from src.utils.config import load_yaml_config
from src.utils.logging_config import configure_logging


def _build_search(
    pipeline: Any,
    config: dict[str, Any],
    random_state: int,
) -> Any:
    search_config = config["training"].get("search", {})
    if not search_config.get("enabled", False):
        return pipeline

    cv = StratifiedKFold(
        n_splits=int(config["training"]["cv_folds"]),
        shuffle=True,
        random_state=random_state,
    )
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_config["param_distributions"],
        n_iter=int(search_config.get("n_iter", 10)),
        scoring=search_config.get("scoring", "f1"),
        n_jobs=int(search_config.get("n_jobs", -1)),
        cv=cv,
        random_state=random_state,
        refit=True,
    )


def train_from_config(config_path: str) -> dict[str, str | float]:
    """Train reproducible NIDS model from YAML configuration."""
    logger = configure_logging("training")
    config = load_yaml_config(config_path)
    random_state = int(config["runtime"]["random_state"])

    dataset = pd.read_csv(config["data"]["path"])
    target = config["data"]["target_column"]
    report = validate_dataset_schema(
        dataset,
        required_columns=config["features"]["numeric"]
        + config["features"]["categorical"],
        target_column=target,
    )
    logger.info(
        "Validated dataset rows=%s cols=%s duplicates=%s",
        report.row_count,
        report.column_count,
        report.duplicate_rows,
    )

    y = dataset[target].astype(int)
    X = dataset.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["runtime"]["test_size"]),
        random_state=random_state,
        stratify=y,
    )

    pipe_config = TrainingPipelineConfig(
        model_name=config["model"]["name"],
        model_params=config["model"]["hyperparameters"],
        random_state=random_state,
        use_smote=bool(config["training"]["use_smote"]),
        smote_params=config["training"].get("smote", {}),
        preprocessing=PreprocessingConfig(
            numeric_features=config["features"]["numeric"],
            categorical_features=config["features"]["categorical"],
        ),
        feature_selection=FeatureSelectionConfig(
            k_best=int(config["training"]["k_best_features"])
        ),
    )

    pipeline = build_training_pipeline(pipe_config)
    cv = StratifiedKFold(
        n_splits=int(config["training"]["cv_folds"]),
        shuffle=True,
        random_state=random_state,
    )
    oof_scores = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    threshold, best_f2 = optimize_f2_threshold(y_train.to_numpy(), oof_scores)

    estimator = _build_search(pipeline, config, random_state)
    estimator.fit(X_train, y_train)
    trained_pipeline = (
        estimator.best_estimator_
        if isinstance(estimator, RandomizedSearchCV)
        else estimator
    )

    test_scores = trained_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_scores >= threshold).astype(int)
    metrics = compute_binary_metrics(y_test.to_numpy(), y_pred, test_scores)
    metrics["threshold"] = float(threshold)
    metrics["best_cv_f2"] = float(best_f2)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    artifact_dir = Path(config["artifacts"]["base_dir"]) / run_id
    metadata = {
        "threshold": threshold,
        "model_version": run_id,
        "features": list(X.columns),
        "config": config,
        "metrics": metrics,
    }
    persist_training_bundle(artifact_dir, trained_pipeline, metadata)

    mlflow_run = log_to_mlflow(
        tracking_uri=config["mlflow"]["tracking_uri"],
        experiment_name=config["mlflow"]["experiment_name"],
        run_name=f"train-{run_id}",
        params={"model": config["model"]["name"], **config["model"]["hyperparameters"]},
        metrics=metrics,
        artifacts_dir=artifact_dir,
        model=trained_pipeline,
    )

    logger.info(
        "Training completed with model_version=%s and mlflow_run=%s",
        run_id,
        mlflow_run,
    )
    return {
        "model_version": run_id,
        "mlflow_run_id": mlflow_run,
        "threshold": float(threshold),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train_from_config(args.config))
