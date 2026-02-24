"""
Training script - CLI entry point for training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.pipelines import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid NIDS model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Override experiment name from config"
    )

    args = parser.parse_args()

    # Run training pipeline
    pipeline = TrainingPipeline(args.config)

    # Override experiment name if provided
    if args.experiment_name:
        pipeline.config["experiment_name"] = args.experiment_name

    experiment_id, metrics = pipeline.run()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nResults saved to: {pipeline.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
