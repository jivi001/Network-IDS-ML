"""
Feedback Buffer — collects SOC analyst labels and triggers retraining.

When a SOC analyst approves, rejects, or relabels an alert in the dashboard,
that feedback is stored here. Once the buffer reaches `trigger_size` samples,
it signals that a retraining cycle should be initiated.

This module is intentionally lightweight (no external DB dependency) —
the buffer is stored as a JSON file so it survives process restarts and
can be committed to DVC for dataset versioning.

Usage:
    buf = FeedbackBuffer(trigger_size=500, buffer_path='feedback.json')
    buf.add(alert_id='abc', original_label='DoS', corrected_label='DDoS',
            features=[...], feedback_type='relabel')
    if buf.should_retrain():
        buf.export_labeled_dataset('feedback_dataset.csv')
        buf.clear()
"""

import json
import csv
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class FeedbackBuffer:
    """
    Persistent buffer for SOC analyst feedback.

    Args:
        trigger_size (int): Number of labeled samples before retraining is triggered.
        buffer_path (str): Path to JSON file where feedback is persisted.
    """

    def __init__(
        self,
        trigger_size: int = 500,
        buffer_path: str = 'feedback_buffer.json',
    ):
        self.trigger_size = trigger_size
        self.buffer_path = buffer_path
        self._buffer: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        alert_id: str,
        feedback_type: str,
        original_label: str,
        features: List[float],
        corrected_label: Optional[str] = None,
        analyst_id: str = 'unknown',
        notes: str = '',
    ) -> None:
        """
        Record a single analyst feedback event.

        Args:
            alert_id:        Unique ID of the alert being reviewed.
            feedback_type:   One of 'approve', 'reject', 'relabel'.
            original_label:  What the model predicted.
            features:        Feature vector of the alert (list of floats).
            corrected_label: True label according to analyst (for 'relabel').
            analyst_id:      Who submitted the feedback.
            notes:           Free-form annotation.
        """
        if feedback_type not in ('approve', 'reject', 'relabel'):
            raise ValueError(
                f"feedback_type must be 'approve', 'reject', or 'relabel', "
                f"got '{feedback_type}'"
            )
        # For 'approve' the corrected label matches the model prediction
        if feedback_type == 'approve':
            corrected_label = corrected_label or original_label
        # 'reject' → false positive → label as Normal
        if feedback_type == 'reject':
            corrected_label = corrected_label or 'Normal'

        entry = {
            'alert_id':        alert_id,
            'feedback_type':   feedback_type,
            'original_label':  original_label,
            'corrected_label': corrected_label,
            'features':        features,
            'analyst_id':      analyst_id,
            'notes':           notes,
            'timestamp':       datetime.now().isoformat(),
        }
        self._buffer.append(entry)
        self._save()
        print(f"[FeedbackBuffer] Added {feedback_type} entry "
              f"(buffer={len(self._buffer)}/{self.trigger_size})")

    def should_retrain(self) -> bool:
        """Returns True when buffer has reached the retraining trigger size."""
        return len(self._buffer) >= self.trigger_size

    def export_labeled_dataset(self, output_path: str) -> int:
        """
        Export feedback buffer as a CSV dataset for retraining.

        Rows: one per feedback entry with features + corrected_label.
        Returns: number of rows exported.
        """
        if not self._buffer:
            print("[FeedbackBuffer] Buffer is empty — nothing to export.")
            return 0

        # Determine number of features from first entry
        n_features = len(self._buffer[0]['features'])
        fieldnames = [f'f{i}' for i in range(n_features)] + ['label']

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._buffer:
                row = {f'f{i}': v for i, v in enumerate(entry['features'])}
                row['label'] = entry['corrected_label']
                writer.writerow(row)

        print(f"[FeedbackBuffer] Exported {len(self._buffer)} samples → {output_path}")
        return len(self._buffer)

    def clear(self) -> None:
        """Clear buffer after successful retraining."""
        n = len(self._buffer)
        self._buffer = []
        self._save()
        print(f"[FeedbackBuffer] Cleared {n} entries after retraining.")

    def summary(self) -> dict:
        """Return summary statistics of the feedback buffer."""
        total = len(self._buffer)
        by_type: Dict[str, int] = {}
        for e in self._buffer:
            by_type[e['feedback_type']] = by_type.get(e['feedback_type'], 0) + 1
        return {
            'total': total,
            'by_type': by_type,
            'trigger_size': self.trigger_size,
            'should_retrain': self.should_retrain(),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        with open(self.buffer_path, 'w') as f:
            json.dump(self._buffer, f, indent=2)

    def _load(self) -> None:
        if os.path.exists(self.buffer_path):
            try:
                with open(self.buffer_path) as f:
                    self._buffer = json.load(f)
                print(f"[FeedbackBuffer] Loaded {len(self._buffer)} entries "
                      f"from {self.buffer_path}")
            except (json.JSONDecodeError, IOError):
                print(f"[FeedbackBuffer] Could not load {self.buffer_path} — "
                      f"starting fresh.")
                self._buffer = []
        else:
            self._buffer = []
