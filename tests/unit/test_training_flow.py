import json
from pathlib import Path

import pandas as pd

from src.training.training import train_from_config


def test_train_from_config_end_to_end(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.csv"
    frame = pd.DataFrame(
        [
            {"n1": 0.1, "c1": "tcp", "label": 0},
            {"n1": 0.2, "c1": "udp", "label": 0},
            {"n1": 1.1, "c1": "tcp", "label": 1},
            {"n1": 1.2, "c1": "udp", "label": 1},
            {"n1": 1.3, "c1": "udp", "label": 1},
            {"n1": 0.3, "c1": "tcp", "label": 0},
        ]
    )
    frame.to_csv(dataset_path, index=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  random_state: 42",
                "  test_size: 0.33",
                "data:",
                f"  path: {dataset_path}",
                "  target_column: label",
                "features:",
                "  numeric: [n1]",
                "  categorical: [c1]",
                "model:",
                "  name: random_forest",
                "  hyperparameters:",
                "    n_estimators: 10",
                "training:",
                "  use_smote: false",
                "  k_best_features: 2",
                "  cv_folds: 2",
                "  search:",
                "    enabled: false",
                "artifacts:",
                f"  base_dir: {tmp_path / 'artifacts'}",
                "mlflow:",
                "  tracking_uri: file:./mlruns",
                "  experiment_name: test-exp",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.training.training.log_to_mlflow", lambda **_: "run-1")
    result = train_from_config(str(cfg_path))
    assert result["mlflow_run_id"] == "run-1"

    artifact_root = tmp_path / "artifacts"
    runs = list(artifact_root.iterdir())
    assert runs
    metadata = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert "threshold" in metadata
