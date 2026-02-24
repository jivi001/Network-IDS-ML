from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_payload(
    frame: pd.DataFrame, expected_features: list[str]
) -> ValidationResult:
    """Validate incoming inference payload schema and enforce feature order."""
    errors: list[str] = []
    missing = [feature for feature in expected_features if feature not in frame.columns]
    extra = [feature for feature in frame.columns if feature not in expected_features]
    if missing:
        errors.append(f"Missing features: {missing}")
    if extra:
        errors.append(f"Unexpected features: {extra}")
    for feature in expected_features:
        if feature in frame.columns and frame[feature].isnull().any():
            errors.append(f"Null values detected in feature '{feature}'")
    return ValidationResult(is_valid=not errors, errors=errors)
