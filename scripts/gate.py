#!/usr/bin/env python
from pathlib import Path
import argparse
import sys
import yaml
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(metrics_path: str, config_path: str) -> bool:
    """Read metrics run and check if it passes minimal MLOps deployment thresholds."""
    metrics_file = Path(metrics_path)
    config_file = Path(config_path)
    
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return False
        
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("thresholds", {})
        
    required_recall = config.get("recall", 0.95)
    required_fpr = config.get("fpr", 0.05)
    required_f1 = config.get("f1_score", 0.90)

    # NIDS Training pipeline outputs tier1 metrics at test_recall, test_fpr, test_f1
    actual_recall = metrics.get("test_recall", metrics.get("recall", 0))
    actual_fpr = metrics.get("test_fpr", metrics.get("fpr", 1.0))
    actual_f1 = metrics.get("test_f1", metrics.get("f1_score", 0))

    passed = True
    logger.info(f"Evaluating metrics from {metrics_path}")
    
    if actual_recall < required_recall:
        logger.error(f"FAIL: Recall {actual_recall:.4f} is below threshold {required_recall:.4f}")
        passed = False
    else:
        logger.info(f"PASS: Recall {actual_recall:.4f} >= {required_recall:.4f}")

    if actual_fpr > required_fpr:
        logger.error(f"FAIL: FPR {actual_fpr:.4f} is above threshold {required_fpr:.4f}")
        passed = False
    else:
        logger.info(f"PASS: FPR {actual_fpr:.4f} <= {required_fpr:.4f}")

    if actual_f1 < required_f1:
        logger.error(f"FAIL: F1 Score {actual_f1:.4f} is below threshold {required_f1:.4f}")
        passed = False
    else:
        logger.info(f"PASS: F1 Score {actual_f1:.4f} >= {required_f1:.4f}")

    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained NIDS model against operational baselines.")
    parser.add_argument("--metrics", required=True, help="Path to the JSON output of the model metrics")
    parser.add_argument("--config", default="configs/training/eval_thresholds.yaml", help="Path to YAML thresholds config")
    args = parser.parse_args()

    success = evaluate_model(args.metrics, args.config)
    if success:
        logger.info("✅ Model passed all evaluation gates. Ready for promotion.")
        sys.exit(0)
    else:
        logger.error("❌ Model failed evaluation gates. Do not deploy.")
        sys.exit(1)
