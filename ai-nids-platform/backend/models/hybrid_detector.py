import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import StackingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

class VAE(nn.Module):
    """Deep Variational Autoencoder for Tier 2 Anomaly Detection."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU()
        )
        self.mu_fc = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_fc(h), self.logvar_fc(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class HybridNIDS:
    """
    Tier 1: Stacking Ensemble (BRF + LightGBM -> Logistic Regression)
    Tier 2: Fusion of Isolation Forest and PyTorch VAE
    """
    def __init__(self, input_dim: int = 49, threshold_percentile: float = 95.0):
        # Tier 1
        brf = BalancedRandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        lgbm = LGBMClassifier(n_estimators=300, is_unbalance=True, random_state=42)
        meta = LogisticRegression(max_iter=1000)
        
        self.tier1_ensemble = StackingClassifier(
            estimators=[('brf', brf), ('lgbm', lgbm)],
            final_estimator=meta,
            passthrough=True,
            cv=5,
            n_jobs=-1
        )
        
        # Tier 2
        self.iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.vae = VAE(input_dim=input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)
        self.vae.eval() # Inference mode by default
        self.threshold_percentile = threshold_percentile
        self.anomaly_threshold = 0.0 # Set during training

    def predict(self, X: np.ndarray, confidence_thresh: float = 0.85):
        """Cascade Inference Logic."""
        n_samples = X.shape[0]
        attack_types = np.empty(n_samples, dtype=object)
        tier_used = np.zeros(n_samples, dtype=int)
        
        # 1. Tier 1 Prediction
        t1_preds = self.tier1_ensemble.predict(X)
        t1_proba = self.tier1_ensemble.predict_proba(X)
        max_conf = np.max(t1_proba, axis=1)
        
        # If Tier 1 is highly confident it's an attack, stop here.
        high_conf_attack = (t1_preds != "Normal") & (max_conf >= confidence_thresh)
        attack_types[high_conf_attack] = t1_preds[high_conf_attack]
        tier_used[high_conf_attack] = 1
        
        # 2. Tier 2 Anomaly Checking (for uncertain or "Normal" traffic)
        tier2_mask = ~high_conf_attack
        if tier2_mask.sum() > 0:
            X_t2 = X[tier2_mask]
            
            # iForest Score (Negative is anomalous)
            if_scores = self.iforest.decision_function(X_t2)
            
            # VAE Score (High MSE is anomalous)
            with torch.no_grad():
                tensor_X = torch.tensor(X_t2, dtype=torch.float32).to(self.device)
                recon, _, _ = self.vae(tensor_X)
                mse_scores = torch.mean((tensor_X - recon)**2, dim=1).cpu().numpy()
            
            # Simple Fusion (In prod, scale and weight these)
            fused_anomaly_score = -if_scores + mse_scores
            
            is_anomaly = fused_anomaly_score > self.anomaly_threshold
            
            t2_labels = np.where(is_anomaly, "Zero_Day_Anomaly", "Normal")
            attack_types[tier2_mask] = t2_labels
            tier_used[tier2_mask] = 2
            
            # We pad anomaly scores for the whole batch for metric tracking
            full_anomaly_scores = np.zeros(n_samples)
            full_anomaly_scores[tier2_mask] = fused_anomaly_score
        else:
            full_anomaly_scores = np.zeros(n_samples)
            
        return {
            "attack_type": attack_types,
            "tier1_proba": t1_proba,
            "anomaly_score": full_anomaly_scores,
            "tier_used": tier_used
        }
