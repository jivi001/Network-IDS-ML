import os

from nids.pipelines.training import TrainingPipeline

# Model Registry Stub
class ModelRegistry:
    def get_production_model(self):
        class DummyProd:
            metrics = {'pr_auc': 0.90}
        return DummyProd()
        
    def promote_to_canary(self, model):
        pass
        
    def flag_failed_training(self, model):
        pass
