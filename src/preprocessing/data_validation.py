from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataValidationReport:
    row_count: int
    column_count: int
    null_counts: dict[str, int]
    duplicate_rows: int


class DataValidationError(ValueError):
    """Raised when dataset fails schema or quality validation."""


def validate_dataset_schema(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    target_column: str,
) -> DataValidationReport:
    """Validate dataframe quality and schema for deterministic training inputs."""
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if target_column not in dataframe.columns:
        raise DataValidationError(f"Missing target column: {target_column}")

    null_counts = dataframe[required_columns + [target_column]].isnull().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        raise DataValidationError(f"Null values found in dataset: {null_counts}")

    duplicate_rows = int(dataframe.duplicated().sum())
    return DataValidationReport(
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        null_counts={k: int(v) for k, v in null_counts.items()},
        duplicate_rows=duplicate_rows,
    )
