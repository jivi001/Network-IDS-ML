import pandas as pd

from nids.preprocessing.preprocessor import NIDSPreprocessor


def test_unseen_categorical_values_fallback_without_missing_in_training():
    """Unseen categories at inference should map to the fallback token safely."""
    train = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 3.0],
            "protocol_type": ["tcp", "udp", "tcp"],
        }
    )

    infer = pd.DataFrame(
        {
            "duration": [4.0],
            "protocol_type": ["icmp"],
        }
    )

    preprocessor = NIDSPreprocessor()
    preprocessor.fit(train)

    transformed = preprocessor.transform(infer)

    assert transformed.shape == (1, 2)
