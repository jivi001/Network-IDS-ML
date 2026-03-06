# Project Overview: Network Intrusion Detection System

## Executive Summary

The Network Intrusion Detection System (NIDS-ML) utilizes an advanced, four-tiered machine learning methodology to detect anomalies, categorize established threat permutations, and maintain sustained precision rates against network structural drift. Its primary focus is accuracy standard retention spanning high-latency constraints necessary for SOC (Security Operations Center) deployments.

## Architectural Tiers and Algorithmic Specifications

### Tier 1: Known Attack Classification via Stacking Ensemble

Network traffic features are assessed for existing threat vector categorizations. The Tier 1 mechanism executes via a robust stacking classification protocol utilizing out-of-fold probabilistic metrics mapped across three separate algorithms:

1. **Balanced Random Forest Classifier**: Employs class-weighted bootstrapping procedures independently scaling the decision boundaries to correctly partition heavily imbalanced node distributions commonly intrinsic to network traffic captures.
2. **Light Gradient Boosting Machine (LGBM)**: Employs leaf-wise mathematical induction and continuous split points to maximize classification rates regarding temporally associated metrics, leveraging localized `is_unbalance` boolean flags for internal minority class maximization.
3. **Calibrated Support Vector Classifier (SVC)**: Utilizes an RBF scaling kernel and calibrated cross-validation mappings to ascertain complex non-linear geometric divisions correlating to specific packet sequence anomalies.

These initial boundary probabilities are structured into secondary variables and passed alongside the original network telemetry to a `LogisticRegression` meta-learner. The output calculates discrete attack categorizations coupled with absolute confidence percentages.

### Tier 2: Zero-Day Anomaly Detection Fusion

When primary evaluation categorizes samples as normal (i.e. outside defined attack parameters bounding limit), the request operates within the secondary unsupervised matrix standard.

1. **Variational Autoencoder (VAE)**: Processes the variables utilizing PyTorch computational distributions mapping input vectors against optimal 16-dimensional embedding bounds, extracting the stochastic mathematical structures characterizing acceptable network behavior. Novel deviations resolve poorly upon neural decoding, generating large mean-squared error (MSE) reconstruction footprints.
2. **Isolation Forest**: Divides spatial dimensions along iterative tree traversals; normal behaviors require mathematically deeper algorithmic partition lines whereas anomalies inherently isolate rapidly.

The ultimate classification resolves utilizing an algorithmic scalar averaging mechanism mapping localized predictions bounded directly by the 95th percentile deviation constants defined during static model instantiation.

### Tier 3: Feature Explainability Layer

To mitigate the inherent opacity of deep learning methodologies, `shap.TreeExplainer` integration resolves the precise mathematical contribution allocated to each specific network variable underlying the diagnostic calculation limit.

### Tier 4: Self-Improving Concept Drift Management

The temporal degradation characteristic of network evolution is resolved via automated observation logic:

1. **ADWIN Concept Drift Protocol**: Extracts the operational stream of prediction error binaries. The sliding data window automatically restricts scale during error stagnation, and dynamically expands during volatile metric alterations. Substantial mathematical differentiation directly signals behavioral pattern modification (concept drift).
2. **Active Learning Selection**: To optimize the cost bounds of subsequent data labeling, continuous probabilistic samples are evaluated by applying Shannon entropy to prediction uncertainty mapping schemas. Uncertain sequences are further isolated using a K-Means geometric diversity filter to select maximally informative boundaries, passing designated anomalies to the internal `FeedbackBuffer` system awaiting SOC verification triggers.

## Performance and Evaluation Methodology

Evaluations enforce cybersecurity prioritization constraints standard specifically to minimizing false negatives.

- **F2-Score Metrics**: Formulated strictly towards weighting detection parameters (Recall) heavily over detection reliability (Precision), optimizing parameters solely to restrict successful threat intrusions.
- **Matthews Correlation Coefficient (MCC)**: Enforces calculation methodologies utilizing true and false classification rates uniformly scaling outputs strictly to a [-1, 1] integer bounds specifying absolute measurement exactitude.
- **Alert Fatigue Index (AFI)**: Operates upon SOC deployment modeling standards calculating total continuous false positive iterations dividing mathematical probability matrices regarding continuous alarm exposure.

Deployments are coupled directly to automated MLOps CI/CD integration standards verifying these discrete conditions algorithmically against pre-commit boundaries utilizing GitHub Actions pipeline limits prior to container deployment protocols executing.
