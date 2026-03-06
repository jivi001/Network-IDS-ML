"""
ADWIN (ADaptive WINdowing) Drift Detector for NIDS.

Wraps `river.drift.ADWIN` to detect concept drift in the model's
per-sample error stream. When drift is detected, the system can:
  1. Log a drift event to MLflow
  2. Trigger the automated retraining pipeline
  3. Reset the window for the next monitoring period

Usage:
    detector = ADWINDriftDetector(delta=0.002)
    for y_true, y_pred in live_stream:
        error = int(y_true != y_pred)
        detector.update(error)
        if detector.drift_detected:
            trigger_retraining()
            detector.reset()

Falls back to a simple Page-Hinkley test if `river` is not installed.
"""

from datetime import datetime
from typing import List, Optional, Callable


# ──────────────────────────────────────────────────────────────────────────────
# ADWIN wrapper
# ──────────────────────────────────────────────────────────────────────────────

try:
    from river import drift as river_drift
    _RIVER_AVAILABLE = True
except ImportError:
    _RIVER_AVAILABLE = False


class ADWINDriftDetector:
    """
    Concept drift detector using ADWIN (Bifet & Gavalda, 2007).

    ADWIN maintains a variable-length sliding window of the error rate
    and signals drift when the error distribution changes significantly.

    Args:
        delta (float): Confidence parameter. Smaller = more sensitive.
                       Typical range: 0.001 (sensitive) – 0.01 (robust).
        on_drift (Callable): Optional callback invoked when drift is detected.
        min_samples (int): Minimum samples before drift can be declared.
    """

    def __init__(
        self,
        delta: float = 0.002,
        on_drift: Optional[Callable] = None,
        min_samples: int = 30,
    ):
        self.delta = delta
        self.on_drift = on_drift
        self.min_samples = min_samples
        self._n_samples = 0
        self._drift_events: List[dict] = []
        self.drift_detected: bool = False

        if _RIVER_AVAILABLE:
            self._detector = river_drift.ADWIN(delta=delta)
            self._mode = 'adwin'
        else:
            # Page-Hinkley fallback
            print("[ADWINDriftDetector] `river` not installed — using "
                  "Page-Hinkley fallback. Install with: pip install river")
            self._ph_sum = 0.0
            self._ph_min = float('inf')
            self._ph_lambda = 50.0   # detection threshold
            self._ph_alpha  = 0.005  # magnitude of acceptable change
            self._mode = 'page_hinkley'

    def update(self, error: int) -> bool:
        """
        Feed one error observation (0 = correct, 1 = wrong prediction).

        Returns:
            True if drift is newly detected, False otherwise.
        """
        self._n_samples += 1
        self.drift_detected = False

        if self._n_samples < self.min_samples:
            return False

        if self._mode == 'adwin':
            self._detector.update(error)
            if self._detector.drift_detected:
                self._handle_drift()
        else:
            self._page_hinkley_update(error)

        return self.drift_detected

    def reset(self) -> None:
        """Reset the detector window after handling a drift event."""
        if self._mode == 'adwin':
            self._detector = type(self._detector)(delta=self.delta)
        else:
            self._ph_sum = 0.0
            self._ph_min = float('inf')
        self.drift_detected = False
        self._n_samples = 0

    @property
    def n_drift_events(self) -> int:
        return len(self._drift_events)

    @property
    def drift_history(self) -> List[dict]:
        return list(self._drift_events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_drift(self) -> None:
        """Record drift event and invoke callback."""
        self.drift_detected = True
        event = {
            'timestamp': datetime.now().isoformat(),
            'n_samples_at_drift': self._n_samples,
            'detector': self._mode,
        }
        self._drift_events.append(event)
        print(f"[ADWINDriftDetector] ⚠️  CONCEPT DRIFT DETECTED at sample "
              f"#{self._n_samples} ({event['timestamp']})")
        if self.on_drift is not None:
            try:
                self.on_drift(event)
            except Exception as exc:
                print(f"[ADWINDriftDetector] on_drift callback failed: {exc}")

    def _page_hinkley_update(self, error: float) -> None:
        """Minimal Page-Hinkley test for drift detection (fallback)."""
        self._ph_sum += error - self._ph_alpha
        self._ph_min = min(self._ph_min, self._ph_sum)
        if self._ph_sum - self._ph_min > self._ph_lambda:
            self._handle_drift()
            self._ph_sum = 0.0
            self._ph_min = float('inf')
