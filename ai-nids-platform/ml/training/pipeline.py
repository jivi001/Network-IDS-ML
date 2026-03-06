import os
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from ai_nids.models.hybrid_detector import HybridNIDS

class TrainingPipeline:
    """Full MLflow integrated training pipeline."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("NIDS-Production-Training")

    def run(self):
        with mlflow.start_run():
            print("1. Loading Data...")
            import polars as pl
            # df = pl.read_csv(self.data_path)
            # X = df.drop("label").to_numpy()
            # y = df["label"].to_numpy()
            # Simulated data for structural demonstration
            import numpy as np
            X = np.random.rand(1000, 49)
            y = np.array(['Normal'] * 800 + ['DoS'] * 150 + ['Probe'] * 50)
            
            print("2. Train/Test Split...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            
            print("3. Handling Imbalance with SMOTE...")
            smote = SMOTE(sampling_strategy='auto')
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            print("4. Training Tier 1 (Stacking Ensemble)...")
            model = HybridNIDS(input_dim=49)
            model.tier1_ensemble.fit(X_train_res, y_train_res)
            
            print("5. Training Tier 2 (Isolation Forest + VAE) on Normal Traffic...")
            normal_mask = (y_train == 'Normal')
            X_normal = X_train[normal_mask]
            model.iforest.fit(X_normal)
            
            # VAE Training Loop (PyTorch)
            import torch
            optimizer = torch.optim.Adam(model.vae.parameters(), lr=1e-3)
            model.vae.train()
            tensor_X = torch.tensor(X_normal, dtype=torch.float32)
            for epoch in range(10): # Short for demo
                optimizer.zero_grad()
                recon, mu, logvar = model.vae(tensor_X)
                recon_loss = torch.nn.functional.mse_loss(recon, tensor_X, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                
            # Adaptive Threshold Calculation
            with torch.no_grad():
                recon_eval, _, _ = model.vae(tensor_X)
                mse_scores = torch.mean((tensor_X - recon_eval)**2, dim=1).numpy()
                if_scores = model.iforest.decision_function(X_normal)
                fused_anomaly_score_train = -if_scores + mse_scores
                model.anomaly_threshold = float(np.percentile(fused_anomaly_score_train, 95))
            print(f"Adaptive Anomaly Threshold Set: {model.anomaly_threshold:.4f}")

            print("6. Evaluation...")
            # Predict
            preds = model.tier1_ensemble.predict(X_test)
            
            # Metrics
            f2 = fbeta_score(y_test == 'Normal', preds == 'Normal', beta=2.0, pos_label=False)
            mlflow.log_metric("f2_score", f2)
            
            print(f"F2 Score (Recall Biased): {f2:.4f}")
            
            print("7. Model Persistence to MLflow Registry...")
            mlflow.sklearn.log_model(model.tier1_ensemble, "tier1_stacking_model")
            
            print("Pipeline Complete.")

if __name__ == "__main__":
    pipeline = TrainingPipeline("data/raw/unsw_nb15.csv")
    pipeline.run()
