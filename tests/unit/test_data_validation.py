import pandas as pd
import pytest

from src.preprocessing import DataValidationError, validate_dataset_schema


def test_validate_dataset_schema_success() -> None:
    frame = pd.DataFrame([{"a": 1, "b": 2, "label": 0}, {"a": 3, "b": 4, "label": 1}])
    report = validate_dataset_schema(
        frame, required_columns=["a", "b"], target_column="label"
    )
    assert report.row_count == 2
    assert report.duplicate_rows == 0


def test_validate_dataset_schema_missing_column() -> None:
    frame = pd.DataFrame([{"a": 1, "label": 0}])
    with pytest.raises(DataValidationError):
        validate_dataset_schema(
            frame, required_columns=["a", "b"], target_column="label"
        )
