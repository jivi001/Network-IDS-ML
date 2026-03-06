from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    dataset_path: str
    target_column: str
    run_name: str
    experiment_name: str
    model_output_dir: str
    mlflow_tracking_uri: str
    test_size: float = 0.2
    random_state: int = 42
    optimize_hyperparameters: bool = True
    optuna_trials: int = 50


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    data = load_yaml(path)
    return RuntimeConfig(**data)

