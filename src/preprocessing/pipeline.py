from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingConfig:
    numeric_features: Sequence[str]
    categorical_features: Sequence[str]


def build_preprocessor(config: PreprocessingConfig) -> ColumnTransformer:
    """Build preprocessor for mixed tabular network telemetry features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(config.numeric_features)),
            ("cat", categorical_pipeline, list(config.categorical_features)),
        ]
    )
