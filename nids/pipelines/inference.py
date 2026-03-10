from pathlib import Path
import numpy as np

from nids.models.unified import UnifiedHybridModel


class InferencePipeline:

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Path to a models/ directory produced by TrainingPipeline,
                       e.g. 'experiments/runs/<experiment_id>/models'
                        or 'models/production/v1.0.0'
        """
        self.model_dir = Path(model_dir)
        self.model = UnifiedHybridModel.load(str(self.model_dir))

    # Add back legacy accessors so explainability or generic code doesn't break entirely if someone depends on it
    @property
    def preprocessor(self):
        return self.model.preprocessor

    @property
    def selector(self):
        return self.model.selector
    
    def predict_single(self, features: np.ndarray) -> dict:
        return self.model.predict(features)
    
    def predict_batch(self, features: np.ndarray) -> dict:
        return self.model.predict(features)