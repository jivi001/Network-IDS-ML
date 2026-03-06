# API Reference: Network Intrusion Detection System

This document specifies the technical boundaries, signatures, and input-output schemas expected across the core NIDS machine learning package and its respective deployment modules.

## 1. Machine Learning Models

### `nids.models.stacking.StackingEnsemble`
Implements the Tier 1 primary supervised classification layer relying on a stacked meta-learner algorithm framework.

- **`__init__(random_state: int = 42, **kwargs)`**
  Initialize the core classifiers (BalancedRandomForest, LGBMClassifier, and CalibratedClassifierCV) with their expected meta-learner (LogisticRegression). Arbitrary keyword specifications define underlying parameters (e.g. `n_estimators_brf`, `svc_c`).
- **`train(X: np.ndarray, y: np.ndarray)`**
  Execute cross-fitting training algorithms. Generates out-of-fold probabilistic maps to inform the secondary meta-learner processing requirements.
- **`predict(X: np.ndarray) -> np.ndarray`**
  Emits designated class estimations following complete algorithm processing.
- **`predict_proba(X: np.ndarray) -> np.ndarray`**
  Outputs direct floating-point probability matrices normalized between 0.0 and 1.0 spanning the categorical parameters.

### `nids.models.anomaly.VAEAnomalyDetector`
Constructs the PyTorch Variational Autoencoder utilized specifically within zero-day anomaly operations.

- **`__init__(n_features: int, latent_dim: int = 16, epochs: int = 50, batch_size: int = 256, learning_rate: float = 1e-3, beta: float = 1.0, threshold_percentile: float = 95.0)`**
  Defines the spatial architecture and iterative processing parameters spanning the encoder and sequential decoding nodes.
- **`train(X: np.ndarray)`**
  Trains exclusively on data specified mathematically as normal traffic mapping variables against optimized multi-dimensional constants. Calculates threshold parameters utilizing empirical error assessments computed post-training.
- **`reconstruction_error(X: np.ndarray) -> np.ndarray`**
  Outputs the absolute scalar difference computed spanning the initial spatial coordinate inputs against decoded variables.
- **`predict(X: np.ndarray) -> np.ndarray`**
  Classifies input objects. Produces continuous vectors mapping directly to integer values `1` (normal) and `-1` (anomaly).

### `nids.models.anomaly.FusionAnomalyDetector`
Manages the integration methodology coupling generic Isolation Forest schemas mathematically mapped to VAE errors.

- **`__init__(vae_params: dict, iforest_params: dict, vae_weight: float = 0.5)`**
  Constructs dual-model instances. Parameters weight prediction calculations specifically against empirical detection values.
- **`train(X: np.ndarray)`**
  Applies simultaneous unsupervised learning parameter allocations against isolation forests and PyTorch dimensions.
- **`predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**
  Requires continuous scalar parameters resolved equally and appended against bounded threshold boundaries to execute positive predictions. Outputs the explicit Boolean evaluation matrix coupled with standardized float scores.

### `nids.models.hybrid.HybridNIDS`
Provides operational cascade processing dictating Tier 1 limits bridging against Tier 2 constraints.

- **`__init__(use_stacking: bool = False, use_vae: bool = False, tier1_params: dict = None, tier2_params: dict = None)`**
  Determines explicit logic mappings for algorithms utilized at prediction inference time. Defaults utilize standard algorithm logic constraints mirroring historical package versions.
- **`predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**
  Translates evaluation matrices internally bounding high-confidence results. Low probability variables trigger sequential anomaly detector bounds. Emits continuous predictive lists spanning both models alongside tier identification limits.
- **`predict_with_scores(X: np.ndarray) -> Dict[str, Any]`**
  Produces detailed JSON schema matrices denoting granular probabilities assigned by all processing systems.

## 2. Infrastructure Optimization and Selection

### `nids.features.selection.FeatureSelector`
Employs specified computational methodology to establish dominant categorical identifiers inherent to intrusion behavior mathematically.

- **`__init__(n_features: int = 20, method: str = 'combined', random_state: int = 42)`**
  Initializes dimension limiting criteria. Supported target string arguments include `importance`, `rfe`, `shap`, `mutual_info`, and `combined`.
- **`fit(X: np.ndarray, y: np.ndarray, feature_names: List[str])`**
  Configures bounds according to the method variables. `'combined'` relies heavily upon structural Borda count calculation frameworks coupling the rankings output by respective methods simultaneously.
- **`transform(X: np.ndarray) -> np.ndarray`**
  Splits and removes irrelevant array components according to trained optimization logic limits.

### `nids.drift.detector.ADWINDriftDetector`
Calculates dynamic network state modification signaling concept drift within deployment patterns.

- **`__init__(delta: float = 0.002, on_drift: Callable = None, min_samples: int = 30)`**
  Establishes standard error-variance tracking mechanisms depending functionally upon the presence of the `river.drift.ADWIN` integration architecture.
- **`update(error: int) -> bool`**
  Requires singular values reflecting predictive inaccuracies mapped against target verification constants. Evaluates immediately against internal temporal windows, broadcasting alerts upon excessive variation bounds.

### `nids.active_learning.query.UncertaintyDiversityQuery`
Extracts maximally important datasets leveraging uncertainty measurements for continued manual refinement.

- **`__init__(budget: int = 50, uncertainty_pool: int = 500)`**
  Configures constraints applying directly to sample sizes returned via inference tracking paths.
- **`select(X_unlabeled: np.ndarray, model: Any) -> np.ndarray`**
  Extrapolates class distributions defining highest structural uncertainty values via strict Shannon entropy computations, sorting limits subsequently via localized spatial boundary algorithms (K-Means) guaranteeing data variation boundaries.

## 3. Evaluation Procedures

### `nids.evaluation.metrics.NIDSEvaluator`
Calculates extensive standard and specialized performance limitations directly correlated to SOC and standard operational thresholds.

- **`evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, labels: List[str] = None, normal_label: str = 'Normal', attack_families: Dict[str, List[str]] = None, detect_latency_ms: float = None) -> Dict[str, float]`**
  Aggregates total matrices formulating categorical performance calculations including precision, recall, F2, MCC, API, and multi-class categorical sub-distributions mapping discrete threat identification statistics.

## 4. API Deployment Reference

The programmatic access variables are bounded inside the FastAPI process container specifications (`deployment/inference_api.py`).

### Health Endpoint
- **Method:** `GET /health`
- **Description:** Outputs explicit integer configurations surrounding operational uptime strings mitigating deployment disruption tracking.

### Predict Endpoints
- **Method:** `POST /predict`
- **Description:** Requires distinct JSON matrices structuring network distribution components. Emits probabilistic evaluation logic spanning explicit threshold limitations. 
- **Method:** `POST /predict/batch`
- **Description:** Allows standard HTTP processing frameworks scaled spanning array indices optimized against sequential matrix evaluations.

### Explainability Endpoint
- **Method:** `POST /explain`
- **Description:** Produces variable array dependencies calculated linearly via SHAP values, denoting granular significance values applying internally to specific array attributes submitted explicitly via the HTTP logic process variables.
