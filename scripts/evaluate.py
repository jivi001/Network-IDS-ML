"""
Evaluation script - CLI entry point for evaluation pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.pipelines import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained NIDS model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to test dataset CSV'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='nsl_kdd',
        choices=['nsl_kdd', 'unsw_nb15', 'cic_ids2017'],
        help='Type of dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for evaluation results'
    )
    
    args = parser.parse_args()
    
    # Run evaluation pipeline
    pipeline = EvaluationPipeline(args.model, args.output)
    metrics = pipeline.run(args.dataset, args.dataset_type)
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"\nResults saved to: {pipeline.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
