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
    parser = argparse.ArgumentParser(description='Train Hybrid NIDS model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Override experiment name from config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=(
            'Override output base directory (sets config[output][base_dir]). '
            'The final artifact directory will be <output>/<experiment_id>/. '
            'Required for CI/CD so gate script finds metrics.json.'
        ),
    )

    args = parser.parse_args()

    # Run training pipeline
    pipeline = TrainingPipeline(args.config)

    # Override experiment name if provided
    if args.experiment_name:
        pipeline.config['experiment_name'] = args.experiment_name

    # FIX: Override output base directory if --output is provided.
    # Without this, `--output experiments/ci_run` was silently ignored because
    # argparse accepted the flag but train.py never used it.  The gate script
    # then searched experiments/ci_run/ and found nothing → CI always failed.
    if args.output:
        pipeline.config.setdefault('output', {})['base_dir'] = args.output
        # Regenerate output_dir so it reflects the new base
        pipeline.output_dir = (
            Path(args.output) / pipeline.experiment_id
        )
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id, metrics = pipeline.run()

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Recall:    {metrics.get('recall', float('nan')):.4f}")
    print(f"Precision: {metrics.get('precision', float('nan')):.4f}")
    print(f"F1-Score:  {metrics.get('f1_score', float('nan')):.4f}")
    print(f"\nResults saved to: {pipeline.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

