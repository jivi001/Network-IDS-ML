# Retraining lifecycle orchestrator

"""
Blueprint for automated model governance and retraining.
Triggered by drift severity or explicit SOC request.
"""

class RetrainingOrchestrator:
    def __init__(self, data_lake_connector, model_registry):
        self.data_lake = data_lake_connector
        self.registry = model_registry
        
    def trigger_retraining(self, trigger_reason: str):
        """Execute closed-loop retraining."""
        # 1. Extraction
        X_train, y_train = self.data_lake.extract_recent_verified_window(days=30)
        
        # 2. Pipeline Rebuild (Ensures stateless build)
        from nids.pipelines import TrainingPipeline
        pipeline = TrainingPipeline()
        model, metrics = pipeline.run(X_train, y_train)
        
        # 3. Model Registry Governance
        current_prod = self.registry.get_production_model()
        
        # 4. Comparative Evaluation
        if metrics['layer1_detection']['pr_auc'] > current_prod.metrics['pr_auc'] * 0.95:
            # Requires approval if metric drops by more than 5%
            self.registry.promote_to_canary(model)
            return {"status": "SUCCESS", "action": "CANARY_DEPLOYED", "reason": trigger_reason}
        else:
            self.registry.flag_failed_training(model)
            return {"status": "FAILED", "action": "ABORTED_DEGRADATION"}
