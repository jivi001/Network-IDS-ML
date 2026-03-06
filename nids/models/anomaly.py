"""
Deep Anomaly Detectors for NIDS Tier 2 — Zero-Day Detection.

Modules:
  VAEAnomalyDetector   — Variational Autoencoder trained on normal traffic only.
                         Zero-day score = reconstruction error (MSE per sample).
  FusionAnomalyDetector — Combines VAE + Isolation Forest scores via weighted average.
                          Returns -1 (anomaly) or 1 (normal) consistent with sklearn API.

Design:
  - VAE is trained ONLY on normal-class samples (same as existing IsolationForest)
  - Threshold is determined from training data reconstruction errors (95th percentile)
  - FusionAnomalyDetector takes an already-trained UnsupervisedModel (IForest) and
    trains a VAE alongside it; scores are fused so neither detector dominates alone
  - PyTorch dependency is optional — falls back to IsolationForest-only if not available

Usage:
    # Stand-alone VAE
    vae = VAEAnomalyDetector(input_dim=20)
    vae.train(X_normal)
    scores = vae.reconstruction_error(X_test)

    # Fusion (VAE + IForest)
    fusion = FusionAnomalyDetector(input_dim=20, vae_weight=0.5)
    fusion.train(X_normal)
    labels = fusion.predict(X_test)   # -1 = anomaly, 1 = normal
"""

import numpy as np
import joblib
from typing import Optional

# IsolationForest as mandatory fallback
from sklearn.ensemble import IsolationForest

# PyTorch optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# VAE building block (only defined when torch is available)
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _VAE(nn.Module):
        """Internal PyTorch VAE module."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.mu_fc = nn.Linear(hidden_dim // 2, latent_dim)
            self.logvar_fc = nn.Linear(hidden_dim // 2, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.mu_fc(h), self.logvar_fc(h)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar

        @staticmethod
        def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
            """ELBO loss: reconstruction + KL divergence."""
            recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + beta * kl


# ──────────────────────────────────────────────────────────────────────────────
# VAEAnomalyDetector
# ──────────────────────────────────────────────────────────────────────────────

class VAEAnomalyDetector:
    """
    Variational Autoencoder anomaly detector.

    Trained on normal-only traffic; anomalies show high reconstruction error.
    Falls back to IsolationForest if PyTorch is not installed.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
        device: str = 'cpu',
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.device_str = device
        self.threshold_: Optional[float] = None
        self.is_trained = False

        torch.manual_seed(random_state) if TORCH_AVAILABLE else None

        if not TORCH_AVAILABLE:
            print("[VAEAnomalyDetector] PyTorch not installed — "
                  "using IsolationForest as fallback.")
            self._fallback = IsolationForest(
                contamination=0.05, random_state=random_state
            )
            self._vae = None
        else:
            self._fallback = None
            self._vae = None

    def train(self, X_normal: np.ndarray) -> None:
        """
        Train VAE on normal-only samples.
        Args:
            X_normal: Feature matrix of normal/benign traffic (n_samples, n_features)
        """
        if not TORCH_AVAILABLE:
            self._fallback.fit(X_normal)
            self.is_trained = True
            print(f"[VAEAnomalyDetector] IsolationForest fallback trained on "
                  f"{X_normal.shape[0]} samples.")
            return

        self.input_dim = self.input_dim or X_normal.shape[1]
        device = torch.device(self.device_str)
        self._vae = _VAE(self.input_dim, self.hidden_dim, self.latent_dim).to(device)
        optimizer = torch.optim.Adam(self._vae.parameters(), lr=self.learning_rate)

        # Normalise to [0,1] using training stats
        self._x_min = X_normal.min(axis=0)
        self._x_range = np.where(
            (X_normal.max(axis=0) - self._x_min) > 0,
            X_normal.max(axis=0) - self._x_min,
            1.0
        )
        X_norm = (X_normal - self._x_min) / self._x_range

        tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._vae.train()
        print(f"[VAEAnomalyDetector] Training VAE ({self.epochs} epochs) on "
              f"{X_normal.shape[0]} normal samples, input_dim={self.input_dim}...")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, mu, logvar = self._vae(batch)
                loss = _VAE.vae_loss(x_hat, batch, mu, logvar, self.beta)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} — loss: {epoch_loss:.4f}")

        # Set anomaly threshold from training reconstruction errors
        train_errors = self._reconstruction_error_internal(X_norm, device)
        self.threshold_ = float(np.percentile(train_errors, self.threshold_percentile))
        self.is_trained = True
        print(f"[VAEAnomalyDetector] Threshold (p{self.threshold_percentile:.0f}): "
              f"{self.threshold_:.6f}")

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample reconstruction MSE (higher = more anomalous)."""
        if not self.is_trained:
            raise RuntimeError("VAEAnomalyDetector must be trained first.")
        if not TORCH_AVAILABLE or self._vae is None:
            # Return negative decision function so higher = more anomalous
            return -self._fallback.decision_function(X)
        X_norm = (X - self._x_min) / self._x_range
        device = torch.device(self.device_str)
        return self._reconstruction_error_internal(X_norm, device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return -1 for anomaly, 1 for normal (sklearn IForest convention)."""
        if not self.is_trained:
            raise RuntimeError("VAEAnomalyDetector must be trained first.")
        if not TORCH_AVAILABLE or self._vae is None:
            return self._fallback.predict(X)
        errors = self.reconstruction_error(X)
        return np.where(errors > self.threshold_, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score (higher = more normal, consistent w/ IForest sign)."""
        return -self.reconstruction_error(X)

    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Must be trained before saving.")
        if TORCH_AVAILABLE and self._vae is not None:
            state = {
                'vae_state': self._vae.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'threshold': self.threshold_,
                'x_min': self._x_min,
                'x_range': self._x_range,
            }
            torch.save(state, filepath)
        else:
            joblib.dump(self._fallback, filepath)
        print(f"[VAEAnomalyDetector] Saved to {filepath}")

    def load(self, filepath: str) -> None:
        if TORCH_AVAILABLE:
            try:
                state = torch.load(filepath, map_location=self.device_str)
                if isinstance(state, dict) and 'vae_state' in state:
                    self.input_dim = state['input_dim']
                    self.hidden_dim = state['hidden_dim']
                    self.latent_dim = state['latent_dim']
                    self._vae = _VAE(
                        self.input_dim, self.hidden_dim, self.latent_dim
                    )
                    self._vae.load_state_dict(state['vae_state'])
                    self._vae.eval()
                    self.threshold_ = state['threshold']
                    self._x_min = state['x_min']
                    self._x_range = state['x_range']
                    self.is_trained = True
                    print(f"[VAEAnomalyDetector] VAE loaded from {filepath}")
                    return
            except Exception:
                pass
        self._fallback = joblib.load(filepath)
        self.is_trained = True
        print(f"[VAEAnomalyDetector] IsolationForest loaded from {filepath}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reconstruction_error_internal(
        self, X_norm: np.ndarray, device
    ) -> np.ndarray:
        self._vae.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
            x_hat, _, _ = self._vae(tensor)
            errors = torch.mean((tensor - x_hat) ** 2, dim=1)
        return errors.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# FusionAnomalyDetector — VAE + IsolationForest score fusion
# ──────────────────────────────────────────────────────────────────────────────

class FusionAnomalyDetector:
    """
    Fusion of VAEAnomalyDetector + IsolationForest for robust zero-day detection.

    Scoring:
        fused_score = vae_weight * vae_normalised + (1 - vae_weight) * iforest_normalised
        anomaly if fused_score > threshold_percentile of training scores
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        vae_weight: float = 0.5,
        iforest_contamination: float = 0.05,
        vae_epochs: int = 50,
        vae_hidden_dim: int = 64,
        vae_latent_dim: int = 16,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
    ):
        assert 0.0 <= vae_weight <= 1.0, "vae_weight must be in [0, 1]"
        self.vae_weight = vae_weight
        self.threshold_percentile = threshold_percentile
        self.threshold_: Optional[float] = None
        self.is_trained = False

        self.vae = VAEAnomalyDetector(
            input_dim=input_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=vae_latent_dim,
            epochs=vae_epochs,
            threshold_percentile=threshold_percentile,
            random_state=random_state,
        )
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=iforest_contamination,
            random_state=random_state,
        )

    def train(self, X_normal: np.ndarray) -> None:
        """Train both sub-detectors on normal traffic."""
        print(f"\n[FusionAnomalyDetector] Training on {X_normal.shape[0]} "
              f"normal samples (VAE weight={self.vae_weight:.2f})...")
        self.vae.train(X_normal)
        self.iforest.fit(X_normal)

        # Compute fused threshold on training data
        fused = self._fused_score(X_normal)
        self.threshold_ = float(np.percentile(fused, self.threshold_percentile))
        self.is_trained = True
        print(f"[FusionAnomalyDetector] Training complete. "
              f"Fusion threshold: {self.threshold_:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return -1 (anomaly) or 1 (normal)."""
        if not self.is_trained:
            raise RuntimeError("FusionAnomalyDetector must be trained first.")
        scores = self._fused_score(X)
        return np.where(scores > self.threshold_, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more normal (negated fused score, consistent w/ IForest API)."""
        if not self.is_trained:
            raise RuntimeError("FusionAnomalyDetector must be trained first.")
        return -self._fused_score(X)

    def save(self, vae_path: str, iforest_path: str) -> None:
        self.vae.save(vae_path)
        joblib.dump(self.iforest, iforest_path)
        print(f"[FusionAnomalyDetector] Saved VAE->{vae_path}, IForest->{iforest_path}")

    def load(self, vae_path: str, iforest_path: str) -> None:
        self.vae.load(vae_path)
        self.iforest = joblib.load(iforest_path)
        self.is_trained = True
        print(f"[FusionAnomalyDetector] Loaded from {vae_path}, {iforest_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fused_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalised fused anomaly score.
        Both sub-scores are mapped to [0,1] before weighting
        (1.0 = most anomalous, 0.0 = most normal).
        """
        vae_raw = self.vae.reconstruction_error(X)
        # IForest decision_function: higher = more normal → invert
        iforest_raw = -self.iforest.decision_function(X)

        vae_norm = _minmax_scale(vae_raw)
        if_norm = _minmax_scale(iforest_raw)

        return self.vae_weight * vae_norm + (1.0 - self.vae_weight) * if_norm


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; returns 0.5 if all values are identical."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)
