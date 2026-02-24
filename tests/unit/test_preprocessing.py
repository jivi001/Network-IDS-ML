import pandas as pd

from src.preprocessing.pipeline import PreprocessingConfig, build_preprocessor


def test_preprocessor_transforms() -> None:
    data = pd.DataFrame(
        [{"num": 1.0, "cat": "a"}, {"num": 2.0, "cat": "b"}, {"num": 3.0, "cat": "a"}]
    )
    pre = build_preprocessor(
        PreprocessingConfig(numeric_features=["num"], categorical_features=["cat"])
    )
    transformed = pre.fit_transform(data)
    assert transformed.shape[0] == 3
