# Methodology: Hybrid Machine Learning-Based Network Intrusion Detection System

## 1. Problem Formulation

### 1.1 Limitations of Signature-Based IDS

Traditional signature-based Intrusion Detection Systems (IDS) rely on predefined attack patterns and rule sets to identify malicious network traffic. While effective against known threats, they exhibit critical limitations in modern threat landscapes:

- **Zero-day vulnerability**: Inability to detect novel or previously unseen attack vectors
- **Signature maintenance overhead**: Continuous manual updates required to maintain detection efficacy
- **Evasion susceptibility**: Attackers can obfuscate payloads to bypass pattern matching
- **High false-negative rates**: Polymorphic and metamorphic malware evade static signatures

### 1.2 Machine Learning-Based IDS Advantages

Machine Learning (ML) approaches address these limitations through:

- **Generalization capability**: Learning underlying attack characteristics rather than exact signatures
- **Adaptive detection**: Continuous model refinement as new attack patterns emerge
- **Anomaly detection**: Identifying deviations from normal behavior without prior attack knowledge
- **Reduced manual intervention**: Automated feature learning and pattern recognition

### 1.3 Research Objective

This research implements a **hybrid two-tier cascade architecture** combining supervised and unsupervised learning to achieve:

1. High-accuracy classification of known attack categories (Tier 1)
2. Zero-day anomaly detection for novel threats (Tier 2)
3. Minimized false negatives through recall-optimized training
4. Practical deployment feasibility with balanced precision-recall trade-offs

---

## 2. Dataset Description and Justification

### 2.1 Selected Datasets

Three benchmark datasets were selected to ensure comprehensive evaluation across diverse attack scenarios:

#### NSL-KDD
- **Origin**: Refined version of KDD Cup 1999 dataset
- **Samples**: 125,973 training records, 22,544 test records
- **Features**: 41 numerical and categorical features
- **Attack Categories**: DoS, Probe, R2L, U2R, Normal
- **Justification**: Widely cited benchmark enabling comparison with existing research; addresses duplicate records issue in original KDD'99

#### UNSW-NB15
- **Origin**: Australian Centre for Cyber Security (2015)
- **Samples**: 2,540,044 records (175,341 training, 82,332 test)
- **Features**: 49 features including flow-based and payload-based attributes
- **Attack Categories**: 9 modern attack families (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)
- **Justification**: Represents contemporary attack vectors absent in legacy datasets; includes hybrid of real normal traffic and synthetic attacks

#### CIC-IDS2017
- **Origin**: Canadian Institute for Cybersecurity (2017)
- **Samples**: 2,830,743 records over 5 days
- **Features**: 80+ features extracted using CICFlowMeter
- **Attack Categories**: Brute Force, DoS/DDoS, Web Attacks, Infiltration, Botnet
- **Justification**: State-of-the-art dataset with realistic network topology; captures full packet payloads and bidirectional flows

### 2.2 Dataset Selection Rationale

The multi-dataset approach ensures:

- **Temporal diversity**: Spanning attack evolution from 1999 to 2017
- **Feature heterogeneity**: Testing model robustness across different feature engineering approaches
- **Attack coverage**: Comprehensive representation of historical and modern threat vectors
- **Generalization validation**: Cross-dataset evaluation prevents overfitting to specific data characteristics

---

## 3. Data Preprocessing Pipeline

### 3.1 Data Cleaning

#### Infinite Value Handling
```
For each feature column f:
    if inf ∈ f:
        f[f == inf] ← max(f[f ≠ inf])
        f[f == -inf] ← min(f[f ≠ -inf])
```
**Rationale**: Preserves distributional extremes while ensuring numerical stability in downstream algorithms.

#### Missing Value Imputation
```
For each feature column f:
    if NaN ∈ f:
        f[f == NaN] ← median(f)
```
**Rationale**: Median imputation robust to outliers; prevents bias introduced by mean imputation in skewed distributions.

### 3.2 Categorical Encoding

**Label Encoding** applied to categorical features (protocol type, service, flag):
```
For each categorical feature c:
    c ← LabelEncoder().fit_transform(c)
```
**Rationale**: Preserves ordinality where applicable; memory-efficient compared to one-hot encoding for high-cardinality features.

**Alternative Considered**: One-hot encoding rejected due to dimensionality explosion (e.g., 70+ service types in NSL-KDD would create 70 binary features).

### 3.3 Feature Scaling

**StandardScaler (Z-score normalization)**:
```
X_scaled = (X - μ) / σ
where μ = mean(X), σ = std(X)
```

**Application**:
- Fitted on training set only
- Transformation applied to both train and test sets using training statistics

**Rationale**:
- Random Forest inherently scale-invariant, but scaling benefits:
  - Isolation Forest (distance-based algorithm)
  - Potential future integration of SVM or neural networks
- Prevents feature dominance based on magnitude rather than information content

### 3.4 Class Imbalance Handling

#### Problem Statement
Network traffic datasets exhibit severe class imbalance:
- Normal traffic: 60-80% of samples
- Minority attacks (e.g., U2R in NSL-KDD): <1% of samples

#### SMOTE (Synthetic Minority Over-sampling Technique)

**Algorithm**:
```
For each minority class sample x_i:
    1. Find k nearest neighbors in feature space
    2. Randomly select neighbor x_nn
    3. Generate synthetic sample:
       x_new = x_i + λ × (x_nn - x_i), λ ∈ [0,1]
```

**Implementation**:
- Applied **only to training set** (prevents data leakage)
- k = 5 neighbors (default)
- Balancing strategy: Over-sample minority classes to match majority class size

**Rationale**:
- Prevents model bias toward majority class
- Improves recall for rare but critical attack types
- Synthetic samples interpolate in feature space rather than duplicating existing records

**Alternative Considered**: Under-sampling rejected due to information loss from discarding majority class samples.

---

## 4. Feature Selection Strategy

### 4.1 Random Forest Feature Importance

**Method**: Gini importance (mean decrease in impurity)
```
For each feature f:
    importance(f) = Σ (weighted impurity decrease across all trees)
```

**Selection Threshold**:
- Rank features by importance scores
- Select top k features capturing 95% cumulative importance
- Typical reduction: 41 features → 20-25 features (NSL-KDD)

### 4.2 Rationale

- **Dimensionality reduction**: Improves computational efficiency and reduces overfitting risk
- **Noise removal**: Eliminates low-information features that may introduce variance
- **Model-specific**: RF importance aligns with Tier 1 classifier, ensuring selected features optimize primary detection layer

### 4.3 Alternative Approaches Considered

- **Recursive Feature Elimination (RFE)**: Computationally expensive for large datasets
- **Correlation-based filtering**: May discard redundant but complementary features
- **Mutual Information**: Requires discretization for continuous features

---

## 5. Model Architecture

### 5.1 Tier 1: Random Forest Classifier

#### Algorithm Overview
Ensemble of decision trees trained on bootstrapped samples with random feature subsets at each split.

#### Hyperparameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_estimators | 100-200 | Balance between performance and computational cost |
| max_depth | 20-30 | Prevents overfitting while capturing complex patterns |
| min_samples_split | 5 | Regularization to avoid leaf nodes with single samples |
| criterion | Gini impurity | Faster computation than entropy with comparable accuracy |
| class_weight | balanced | Addresses residual imbalance post-SMOTE |

#### Training Data
- SMOTE-balanced training set
- All attack categories + Normal class
- Feature-selected subset (top k features)

#### Output
- Multi-class predictions: {Normal, DoS, Probe, R2L, U2R, ...}
- Class probability distributions for confidence scoring

### 5.2 Tier 2: Isolation Forest (Anomaly Detector)

#### Algorithm Overview
Unsupervised ensemble method isolating anomalies through random partitioning. Anomalies require fewer splits to isolate due to their rarity and distinctiveness.

#### Hyperparameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_estimators | 100-150 | Sufficient trees for stable anomaly score convergence |
| contamination | 0.05-0.10 | Expected proportion of anomalies in "Normal" traffic |
| max_samples | 256 | Subsample size for computational efficiency |
| random_state | 42 | Reproducibility |

#### Training Data
- **Normal traffic only** (extracted from training set)
- Rationale: Models "normal" behavior distribution; deviations flagged as anomalies

#### Output
- Binary predictions: {1 (normal), -1 (anomaly)}
- Anomaly scores (negative values indicate anomalies)

### 5.3 Baseline: Support Vector Machine (SVM)

#### Purpose
Comparative baseline to validate RF performance.

#### Configuration
- Kernel: Radial Basis Function (RBF)
- C (regularization): 1.0
- Gamma: scale (1 / (n_features × X.var()))

#### Evaluation Scope
- Trained on same SMOTE-balanced data
- Evaluated on identical test set
- Metrics compared: Precision, Recall, F1-Score, training time

---

## 6. Hybrid Detection Workflow

### 6.1 Cascade Architecture

```
┌─────────────────┐
│ Network Traffic │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ← Cleaning, Encoding, Scaling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Select  │ ← Top k features by RF importance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TIER 1: RF      │ ← Multi-class attack classification
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Attack │ │ Normal │
└────────┘ └───┬────┘
    │          │
    │          ▼
    │     ┌─────────────────┐
    │     │ TIER 2: iForest │ ← Anomaly detection
    │     └────────┬────────┘
    │              │
    │         ┌────┴────┐
    │         │         │
    │         ▼         ▼
    │    ┌─────────┐ ┌────────┐
    │    │ Anomaly │ │ Normal │
    │    └─────────┘ └────────┘
    │         │         │
    ▼         ▼         ▼
┌──────────────────────────┐
│   Alert / Pass Decision  │
└──────────────────────────┘
```

### 6.2 Cascade Logic

**Step 1**: All traffic processed by Tier 1 (Random Forest)
- If prediction ≠ Normal → **Immediate Alert** (known attack detected)
- If prediction = Normal → Forward to Tier 2

**Step 2**: Tier-1-Normal traffic processed by Tier 2 (Isolation Forest)
- If anomaly score < threshold → **Zero-Day Alert** (novel anomaly detected)
- If anomaly score ≥ threshold → **Pass** (benign traffic)

### 6.3 Design Rationale

**Advantages**:
- **Reduced false positives**: Tier 2 only processes RF-Normal traffic, avoiding redundant anomaly flags on already-classified attacks
- **Computational efficiency**: Isolation Forest (computationally expensive) processes reduced sample set
- **Complementary coverage**: Supervised layer handles known attacks; unsupervised layer catches zero-days

**Trade-offs**:
- **Tier 1 dependency**: Tier 2 never sees samples misclassified as attacks by RF (potential false negatives if RF fails)
- **Mitigation**: High-recall RF training minimizes this risk

---

## 7. Evaluation Metrics and Rationale

### 7.1 Primary Metrics

#### Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)
```
**Priority**: **Highest** in security context
**Rationale**: Minimizing False Negatives (missed attacks) is critical; undetected intrusions pose greater risk than false alarms.

#### Precision (Positive Predictive Value)
```
Precision = TP / (TP + FP)
```
**Rationale**: Balances alert fatigue; excessive false positives overwhelm SOC analysts and reduce trust in system.

#### F1-Score (Harmonic Mean)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Rationale**: Single metric balancing precision-recall trade-off; useful for model comparison.

### 7.2 Secondary Metrics

#### Confusion Matrix
Per-class breakdown of TP, TN, FP, FN for multi-class evaluation.

#### Precision-Recall Curve (PR-AUC)
**Preference over ROC-AUC**: PR curves better suited for imbalanced datasets; ROC curves can be overly optimistic when negative class dominates.

#### Weighted Averages
Macro-average (equal weight per class) vs. weighted-average (proportional to class support) to assess performance across imbalanced classes.

### 7.3 Tier-Specific Evaluation

**Tier 1 (Random Forest)**:
- Multi-class classification metrics
- Per-attack-category recall (ensure minority attacks not ignored)

**Tier 2 (Isolation Forest)**:
- Binary classification (anomaly vs. normal)
- Evaluated only on Tier-1-Normal subset
- Contamination parameter tuned via validation set

**Hybrid System**:
- End-to-end metrics on full test set
- Tier activation statistics (% samples processed by Tier 2)

---

## 8. Limitations and Assumptions

### 8.1 Limitations

**Dataset Constraints**:
- Benchmark datasets may not reflect current threat landscape (e.g., NSL-KDD from 1999)
- Synthetic attack generation may not capture real-world attack complexity

**Model Constraints**:
- Random Forest: Computationally expensive for real-time inference on high-throughput networks
- Isolation Forest: Contamination parameter requires domain knowledge; misestimation degrades performance

**Evaluation Constraints**:
- Static test sets do not simulate concept drift over time
- Cross-dataset generalization not extensively validated

### 8.2 Assumptions

**Data Assumptions**:
- Training data representative of operational network traffic distribution
- Attack labels accurately annotated (ground truth reliability)

**Deployment Assumptions**:
- Preprocessing pipeline (feature extraction) aligned with training schema
- Network topology and traffic patterns remain relatively stable

**Threat Model Assumptions**:
- Adversaries do not have white-box access to model (no adversarial evasion attacks)
- Zero-day attacks exhibit distributional deviations detectable by Isolation Forest

---

## 9. Scalability and Concept Drift

### 9.1 Scalability Considerations

**Computational Complexity**:
- Random Forest inference: O(n_trees × log(n_samples))
- Isolation Forest: O(n_trees × max_samples)
- Bottleneck: Feature extraction and preprocessing for high-speed networks (>10 Gbps)

**Mitigation Strategies**:
- **Model compression**: Reduce tree depth and ensemble size post-training
- **Parallel processing**: Distribute inference across multiple cores/nodes
- **Feature caching**: Precompute flow-based features for session-level analysis

### 9.2 Concept Drift Handling

**Problem**: Network traffic patterns and attack strategies evolve over time, degrading model performance.

**Proposed Solutions**:
- **Periodic retraining**: Schedule model updates (e.g., monthly) with recent labeled data
- **Online learning**: Incremental updates using streaming algorithms (future work)
- **Drift detection**: Monitor performance metrics (e.g., rolling recall); trigger retraining when degradation detected
- **Ensemble diversity**: Maintain multiple models trained on different time windows; aggregate predictions

**Tier 2 Advantage**: Isolation Forest requires only normal traffic for retraining, reducing labeling overhead.

---

## 10. Summary

This methodology presents a hybrid ML-based NIDS combining Random Forest (supervised) and Isolation Forest (unsupervised) in a cascade architecture. The approach addresses class imbalance via SMOTE, prioritizes recall to minimize false negatives, and provides zero-day detection capability. Evaluation across three benchmark datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017) ensures robustness across diverse attack scenarios. While limitations exist regarding real-time scalability and concept drift, the modular design enables iterative refinement and operational deployment in security operations centers.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Target Audience**: Academic reviewers, engineering faculty, security researchers
