"""
Cross-dataset evaluation script.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.evaluation import CrossDatasetEvaluator


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument(
        "--source-model", type=str, required=True, help="Path to source model directory"
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        required=True,
        help="Name of source dataset (e.g., nsl_kdd)",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        required=True,
        help="Name of target dataset (e.g., unsw_nb15)",
    )
    parser.add_argument(
        "--target-data", type=str, required=True, help="Path to target dataset CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/cross_dataset",
        help="Output directory",
    )

    args = parser.parse_args()

    # Run cross-dataset evaluation
    evaluator = CrossDatasetEvaluator(args.source_model, args.source_dataset)
    report = evaluator.evaluate_on_target(
        args.target_dataset, args.target_data, args.output
    )

    print(f"\n{'='*70}")
    print(f"Cross-Dataset Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Source: {args.source_dataset}")
    print(f"Target: {args.target_dataset}")
    print(f"Recall: {report['metrics']['recall']:.4f}")
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Common features: {report['alignment']['common_features']}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
