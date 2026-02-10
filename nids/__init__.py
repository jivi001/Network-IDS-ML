"""
NIDS - Network Intrusion Detection System

A production-grade hybrid ML-based intrusion detection system combining:
- Random Forest (supervised detection of known attacks)
- Isolation Forest (unsupervised zero-day anomaly detection)

Supports NSL-KDD, UNSW-NB15, and CIC-IDS2017 datasets.
"""

__version__ = "1.0.0"
__author__ = "Network-IDS-ML Team"

from nids.data import DataLoader
from nids.preprocessing import NIDSPreprocessor
from nids.features import FeatureSelector
from nids.models import HybridNIDS, SupervisedModel, UnsupervisedModel
from nids.evaluation import NIDSEvaluator

__all__ = [
    "DataLoader",
    "NIDSPreprocessor",
    "FeatureSelector",
    "HybridNIDS",
    "SupervisedModel",
    "UnsupervisedModel",
    "NIDSEvaluator",
]
