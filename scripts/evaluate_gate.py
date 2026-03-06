#!/usr/bin/env python3
"""
evaluate_gate.py — Model Quality Gate for CI/CD Pipeline.

Reads a metrics.json from the latest experiment run and checks whether
the model passes the minimum threshold on PR-AUC, F2, and MCC.

Exits:
    0  — all gates passed (CI continues to Docker build + deploy)
    1  — one or more gates failed (CI blocks deployment, raises :error:)

Usage:
    python scripts/evaluate_gate.py \\
        --metrics-dir experiments/ci_run \\
        --min-pr-auc 0.90 \\
        --min-f2 0.90 \\
        --min-mcc 0.85 \\
        --output gate_result.json
"""

import argparse
import json
import sys
from pathlib import Path


def find_metrics(metrics_dir: str) -> dict:
    """
    Search for metrics.json within the experiment directory.
    Supports both flat and nested experiment structures.
    """
    base = Path(metrics_dir)
    candidates = list(base.rglob("metrics.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.json found under '{metrics_dir}'. "
            "Run train.py first."
        )
    # Use the most recently modified one
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[evaluate_gate] Reading metrics from: {latest}")
    with open(latest) as f:
        return json.load(f)


def run_gate(
    metrics: dict,
    min_pr_auc: float,
    min_f2: float,
    min_mcc: float,
) -> dict:
    """
    Compare each metric against its gate threshold.
    Returns a structured result dict.
    """
    checks = {
        "pr_auc": {
            "value":     metrics.get("pr_auc"),
            "threshold": min_pr_auc,
            "pass":      None,
        },
        "f2_score": {
            "value":     metrics.get("f2_score"),
            "threshold": min_f2,
            "pass":      None,
        },
        "mcc": {
            "value":     metrics.get("mcc"),
            "threshold": min_mcc,
            "pass":      None,
        },
    }

    rows = []
    all_pass = True
    for name, check in checks.items():
        val = check["value"]
        thr = check["threshold"]
        if val is None:
            # Metric not present — treat as failed with helpful message
            check["pass"] = False
            all_pass = False
            status = "MISSING"
        else:
            check["pass"] = float(val) >= thr
            if not check["pass"]:
                all_pass = False
            status = "✅ PASS" if check["pass"] else "❌ FAIL"
        rows.append((name, val, thr, status))

    # Print table
    print(f"\n{'Metric':<15} {'Value':>8}  {'Threshold':>10}  {'Status'}")
    print("-" * 50)
    for name, val, thr, status in rows:
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"{name:<15} {val_str:>8}  {thr:>10.4f}  {status}")
    print()

    return {
        "pass":    all_pass,
        "checks":  checks,
        "summary": "All gates passed." if all_pass else "One or more gates FAILED.",
    }


def main():
    parser = argparse.ArgumentParser(description="NIDS Model Quality Gate")
    parser.add_argument(
        "--metrics-dir", required=True,
        help="Directory containing metrics.json (searched recursively)"
    )
    parser.add_argument("--min-pr-auc", type=float, default=0.90)
    parser.add_argument("--min-f2",     type=float, default=0.90)
    parser.add_argument("--min-mcc",    type=float, default=0.85)
    parser.add_argument("--output",     default="gate_result.json",
                        help="Path to write gate result JSON")
    args = parser.parse_args()

    try:
        metrics = find_metrics(args.metrics_dir)
    except FileNotFoundError as e:
        print(f"[evaluate_gate] ERROR: {e}")
        result = {"pass": False, "error": str(e)}
        Path(args.output).write_text(json.dumps(result, indent=2))
        sys.exit(1)

    result = run_gate(
        metrics,
        min_pr_auc=args.min_pr_auc,
        min_f2=args.min_f2,
        min_mcc=args.min_mcc,
    )

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"[evaluate_gate] Result written to {args.output}")
    print(f"[evaluate_gate] Gate {'PASSED ✅' if result['pass'] else 'FAILED ❌'}")

    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
