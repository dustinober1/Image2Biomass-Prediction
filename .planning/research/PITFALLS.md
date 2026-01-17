# Domain Pitfalls

**Domain:** ML Experimentation Frameworks
**Researched:** 2025-01-17
**Confidence:** HIGH (verified with scikit-learn official docs and WebSearch 2024)

## Critical Pitfalls

Mistakes that cause rewrites, failed experiments, or irreproducible research.

### Pitfall 1: Data Leakage Through Improper Preprocessing

**What goes wrong:** Information from test/validation sets leaks into training via preprocessing steps like scaling, imputation, or feature selection. This creates artificially high performance that doesn't generalize.

**Why it happens:**
- Applying `StandardScaler.fit()` on entire dataset before train/test split
- Feature selection performed on full dataset instead of training-only
- Target encoding calculated using all data
- Imputation using global statistics (mean/median from all data)

**Consequences:**
- Overly optimistic metrics (e.g., 0.99 R² on random data)
- Models fail catastrophically in production
- Wasted time chasing "improvements" that are measurement errors
- Irreproducible results across different data splits

**Prevention:**
```python
# WRONG - leakage
scaler = StandardScaler().fit(X)  # All data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RIGHT - no leakage
scaler = StandardScaler().fit(X_train)  # Training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# BETTER - use Pipeline (recommended by scikit-learn)
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), RandomForestRegressor())
model.fit(X_train, y_train)
```

**Detection:**
- Performance is suspiciously high (near-perfect on random labels)
- Large gap between cross-validation and hold-out test performance
- Feature importance shows features that shouldn't be predictive
- Results change dramatically when using proper pipelines

**Phase to address:** Foundation (Phase 1-2). Build framework with pipelines from day one.

---

### Pitfall 2: Inconsistent Random State Management

**What goes wrong:** Random seeds not properly controlled, leading to irreproducible experiments. Using `RandomState` instances vs integers causes different CV behavior.

**Why it happens:**
- Using `random_state=None` (default) varies across runs
- Passing `np.random.RandomState(0)` to estimators vs `random_state=0` changes CV behavior
- Not setting seeds in all randomness sources (NumPy, TensorFlow, PyTorch)
- Random state mutated by multiple calls to `fit()`

**Consequences:**
- Cannot reproduce "good" results
- Cross-validation scores vary inconsistently
- Fair comparison impossible (different models evaluated on different splits)
- "It worked yesterday but not today"

**Prevention:**
```python
# For reproducible runs (same results every execution)
rng = np.random.RandomState(42)
X, y = make_classification(random_state=rng)
model = RandomForestClassifier(random_state=rng)
X_train, X_test = train_test_split(X, y, random_state=rng)

# For robust CV (different randomness each fold - preferred)
model = RandomForestClassifier(random_state=None)  # Varies per fold
cv = KFold(shuffle=True, random_state=42)  # Same splits each run

# NEVER mix these:
rf_wrong = RandomForestClassifier(random_state=np.random.RandomState(0))
# This changes behavior across CV folds, making comparison unfair
```

**Detection:**
- Running same script twice gives different results
- Team members get different scores on same code
- CV scores are inconsistent when rerunning

**Phase to address:** Foundation (Phase 1). Establish seeding protocol before running experiments.

---

### Pitfall 3: P-Hacking Through Repeated Validation Set Use

**What goes wrong:** Repeatedly evaluating on validation set while tuning hyperparameters, effectively "training" on it. Also called "data dredging" or "significance chasing."

**Why it happens:**
- No strict hold-out test set separated from validation
- Using validation set for both hyperparameter tuning AND model selection
- Reporting best validation score across many runs without statistical correction
- Tuning hyperparameters based on test set performance

**Consequences:**
- Overfitting to validation set
- Published results don't replicate
- "We achieved 95% accuracy" becomes 75% in production
- Research contributes to reproducibility crisis

**Prevention:**
```python
# Three-way split: train | validation | test
# Train: fit model parameters
# Validation: tune hyperparameters (early stopping, regularization)
# Test: FINAL evaluation only, never used for decisions

# WRONG - p-hacking
for lr in [0.001, 0.01, 0.1]:
    model.train(X_train, y_train, lr=lr)
    score = model.evaluate(X_test, y_test)  # Using test for tuning!
    if score > best_score:
        best_lr = lr

# RIGHT - proper separation
for lr in [0.001, 0.01, 0.1]:
    model.train(X_train, y_train, lr=lr)
    score = model.evaluate(X_val, y_val)  # Tune on validation
    if score > best_score:
        best_lr = lr

# Final evaluation ONCE
final_model.train(X_train, y_train, lr=best_lr)
final_score = final_model.evaluate(X_test, y_test)  # Never tune on this
```

**Detection:**
- Hyperparameters chosen based on test set performance
- Test set used more than once
- Validation score suspiciously close to test score
- Many experiments run but only best reported

**Phase to address:** Experiment Design (Phase 2). Establish data split protocol before first experiment.

---

### Pitfall 4: Cherry-Picking Results

**What goes wrong:** Selectively reporting only successful experiments, datasets, or metrics. Hiding failed experiments creates inflated perception of performance.

**Why it happens:**
- Pressure to publish positive results
- Failed experiments seen as "wasted time"
- No systematic tracking of all experiments
- Reporting best fold from cross-validation instead of mean

**Consequences:**
- Methods appear better than they are
- Time wasted by others trying to replicate
- Overestimation of approach effectiveness
- Research not reproducible

**Prevention:**
```python
# WRONG - cherry-picking
experiments = [run_exp(config) for config in configs]
best_score = max(e.score for e in experiments)
print(f"Best result: {best_score}")  # Only reports winner

# RIGHT - report all results
for exp in experiments:
    log_experiment(exp.config, exp.score, exp.metrics)
print(f"Mean: {mean(e.score for e in experiments)} ± {std(...)}")
print(f"Best: {max(...)}, Worst: {min(...)}")
```

**Detection:**
- Paper/project reports only best results
- No mention of failed experiments
- Error bars missing from reported metrics
- Selecting specific datasets that favor the method

**Phase to address:** Experiment Tracking (Phase 2-3). Build framework that automatically logs ALL experiments.

---

### Pitfall 5: Metric Gaming and Misleading Evaluation

**What goes wrong:** Optimizing for metrics that don't reflect real-world goals, or using metrics that hide poor performance.

**Why it happens:**
- Using accuracy on imbalanced datasets (99% accuracy = predicting majority class)
- R² score can be artificially inflated by wrong preprocessing
- Not evaluating on all relevant subsets (e.g., only overall RMSE, not per-class)
- Selecting metrics after seeing results

**Consequences:**
- Model looks good but fails in practice
- Poor performance on important subgroups masked
- Optimizing wrong objective
- Deployment failures

**Prevention:**
```python
# WRONG - misleading metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # 99% on imbalanced data
# But model just predicts majority class

# RIGHT - comprehensive evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")
print(f"F1-score: {f1_score(y_test, y_pred)}")
print(f"Per-class recall: {classification_report(y_test, y_pred)}")
print(f"RMSE: {rmse}, MAE: {mae}")  # Multiple metrics

# Evaluate on subsets
for biomass_type in ['dry', 'fresh', 'aboveground']:
    mask = y_test['type'] == biomass_type
    print(f"{biomass_type} RMSE: {rmse(y_test[mask], y_pred[mask])}")
```

**Detection:**
- Only one metric reported
- Metric doesn't match business goal (e.g., accuracy for rare event prediction)
- No evaluation on important subgroups
- Test set leakage through metric selection

**Phase to address:** Evaluation Framework (Phase 3). Build comprehensive evaluation suite upfront.

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or confusion but not complete failure.

### Pitfall 6: Cross-Validation Fold Inconsistency

**What goes wrong:** When comparing models, using CV splitters with `RandomState` instances causes different splits for each model. Makes fold-to-fold comparison invalid.

**Why it happens:**
```python
# Wrong - different splits for each model
cv = KFold(shuffle=True, random_state=np.random.RandomState(42))
for model in [rf, svm, knn]:
    scores = cross_val_score(model, X, y, cv=cv)  # Different splits each call!
    # Cannot compare fold 1 of rf with fold 1 of svm

# Right - same splits
cv = KFold(shuffle=True, random_state=42)  # Integer
for model in [rf, svm, knn]:
    scores = cross_val_score(model, X, y, cv=cv)  # Same splits
```

**Consequences:**
- Unfair model comparison
- Higher variance in differences than expected
- Confusion about which model is better

**Prevention:** Always pass integers to CV splitters when comparing models.

**Detection:** Fold-to-fold scores vary wildly between model comparisons.

**Phase to address:** Experiment Design (Phase 2).

---

### Pitfall 7: Environment Drift

**What goes wrong:** Experiment results change over time as dependencies update (NumPy, scikit-learn, CUDA).

**Why it happens:**
- No frozen requirements.txt
- Random number generation algorithms change
- Default algorithm implementations change
- CUDA/cuDNN versions affect GPU training

**Consequences:**
- Cannot reproduce results from 6 months ago
- "It worked on my laptop but not the server"
- Paper results cannot be replicated

**Prevention:**
```python
# Freeze exact versions
pip freeze > requirements.txt

# Better: use conda-lock or pip-tools
# Best: use Docker containers

# Record environment details
import numpy, sklearn, torch
print(f"NumPy: {numpy.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

**Detection:**
- Results change after `pip install -U`
- Different results on different machines
- Old experiments don't match current runs

**Phase to address:** Infrastructure (Phase 1). Set up reproducible environment from day one.

---

### Pitfall 8: Inconsistent Data Splits

**What goes wrong:** Train/test splits vary across experiments, making comparison impossible.

**Why it happens:**
- Splitting data inside each experiment script
- Using different random states
- No canonical train/test split saved
- Data preprocessing changes samples

**Consequences:**
- Cannot fairly compare experiments
- "Which experiment is better?" is unanswerable
- Wasted time re-running experiments

**Prevention:**
```python
# Split once, save, reuse
import json
split_file = "data_splits.json"

if not exists(split_file):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    with open(split_file, 'w') as f:
        json.dump({
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist()
        }, f)

# All experiments use same split
with open(split_file) as f:
    splits = json.load(f)
    X_train, X_test = X[splits['train_indices']], X[splits['test_indices']]
```

**Detection:**
- Different experiments report different test scores
- No canonical split file exists
- Test set changes between runs

**Phase to address:** Data Management (Phase 1). Create canonical splits before first experiment.

---

### Pitfall 9: Not Tracking Experiment Metadata

**What goes wrong:** Running experiments but not recording hyperparameters, data versions, code versions, or environment details.

**Why it happens:**
- Ad-hoc scripts without logging
- "I'll remember these settings"
- No experiment tracking framework
- Results scattered across notebooks

**Consequences:**
- "What hyperparameters did we use for that 0.85 score?"
- Cannot reproduce past experiments
- Lost work when laptop crashes
- No systematic learning from failures

**Prevention:**
```python
# Use MLflow, Weights & Biases, or simple JSON logging
import json
from datetime import datetime

experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'model': 'RandomForestRegressor',
    'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
    'data_version': 'v1.2',  # Track data changes
    'git_commit': 'abc123',  # Track code changes
    'metrics': {'rmse': 0.45, 'r2': 0.82},
    'random_seed': 42
}
with open(f'experiments/{timestamp}.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

**Detection:**
- Questions like "what were the settings for exp-7?"
- No record of failed experiments
- Cannot explain why one experiment worked better

**Phase to address:** Experiment Tracking (Phase 2-3). Build logging into framework from start.

---

### Pitfall 10: Small Dataset Overfitting

**What goes wrong:** With small datasets (357 images in this project), models can easily overfit, and cross-validation variance is high.

**Why it happens:**
- High model complexity relative to data
- Not using proper cross-validation
- Data augmentation issues
- Over-reliance on single train/test split

**Consequences:**
- Model appears to perform well but fails on new data
- High variance in performance across different random splits
- Overfitting to specific test set

**Prevention:**
```python
# Use stratified k-fold cross-validation
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Report mean and std
scores = cross_val_score(model, X, y, cv=cv)
print(f"Score: {scores.mean():.3f} ± {scores.std():.3f}")

# Use simpler models as baseline
# Consider regularization
# Use nested CV for hyperparameter tuning
```

**Detection:**
- Large standard deviation across CV folds
- Performance drops significantly on different random splits
- Complex model beats simple model by small margin

**Phase to address:** Model Selection (Phase 3-4). Use nested CV for small datasets.

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 11: Not Setting Baselines

**What goes wrong:** Jumping to complex models without establishing simple baselines.

**Prevention:** Always start with:
- Mean prediction
- Linear regression
- Random forest with default params

**Phase to address:** Model Selection (Phase 3).

---

### Pitfall 12: Ignoring Data Distribution Shift

**What goes wrong:** Not checking if train/test sets come from same distribution.

**Prevention:**
```python
# Compare distributions
print(f"Train mean: {y_train.mean()}, Test mean: {y_test.mean()}")
print(f"Train std: {y_train.std()}, Test std: {y_test.std()}")

# Use statistical tests
from scipy.stats import ks_2samp
stat, pval = ks_2samp(y_train, y_test)
print(f"KS test p-value: {pval}")
```

**Phase to address:** Data Analysis (Phase 1).

---

### Pitfall 13: Reproducibility vs Robustness Trade-off

**What goes wrong:** Using `random_state=42` everywhere gives reproducible CV, but doesn't test model robustness to randomness.

**Prevention:**
- Use `random_state=42` for reproducible runs
- Use `random_state=None` for robust CV evaluation
- Report both

**Phase to address:** Experiment Design (Phase 2).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **Phase 1: Data Management** | Inconsistent data splits, Environment drift | Create canonical splits, freeze requirements |
| **Phase 2: Experiment Design** | P-hacking, Random state issues | Three-way split, seeding protocol |
| **Phase 3: Experiment Tracking** | Cherry-picking, Missing metadata | Auto-log all experiments, even failures |
| **Phase 4: Model Evaluation** | Metric gaming, Data leakage | Comprehensive eval suite, use pipelines |
| **Phase 5: Analysis** | Overfitting to small data, No statistical tests | Nested CV, report confidence intervals |
| **Phase 6: Reporting** | Selective reporting, Missing error bars | Report all results, use statistical significance tests |

---

## Statistical Rigor Checklist

For each experiment, ensure:

- [ ] Data leakage prevented (pipelines used)
- [ ] Proper train/validation/test split maintained
- [ ] Random seeds recorded and consistent
- [ ] Test set used only for final evaluation
- [ ] Multiple metrics reported (not just accuracy/R²)
- [ ] Confidence intervals or standard deviations reported
- [ ] Failed experiments logged and analyzed
- [ ] Environment details recorded (package versions)
- [ ] Data version tracked
- [ ] Code version tracked (git commit)
- [ ] Results reproducible by teammate
- [ ] Statistical significance tested for improvements

---

## Sources

### HIGH Confidence (Official Documentation)
- **scikit-learn Common Pitfalls**: https://scikit-learn.org/stable/common_pitfalls.html
  - Data leakage prevention
  - Random state management
  - Pipeline usage
  - Cross-validation best practices

### MEDIUM Confidence (Verified WebSearch 2024)
- **ML Research Statistical Errors 2024**: princeton.edu, arxiv.org
  - P-hacking in AI research
  - Cherry-picking and selective reporting
  - Reproducibility crisis factors
  - Data leakage as major error source

- **Data Leakage Pitfalls 2024**: shelf.io, medium.com, ibm.com, yale.edu
  - Global scaling leakage
  - Improper cross-validation
  - Feature engineering leakage
  - Detection red flags

- **ML Experiment Comparison 2024**: lumenova.ai, mdpi.com, brookings.edu
  - Fairness metrics for evaluation
  - Bias mitigation strategies
  - Comparison methodology

### LOW Confidence (WebSearch only, needs verification)
- **ML Experiment Management**: Scalability issues (API quota exhausted before verification)
- **Framework-specific patterns**: MLflow, W&B best practices (access denied)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Data Leakage | HIGH | Verified with scikit-learn official docs |
| Random State Management | HIGH | Verified with scikit-learn official docs |
| P-Hacking | MEDIUM | WebSearch 2024, multiple sources agree |
| Cherry-Picking | MEDIUM | WebSearch 2024, arxiv.org verification |
| Statistical Rigor | MEDIUM | WebSearch 2024, Princeton/arxiv sources |
| Environment Drift | MEDIUM | Well-known issue, multiple sources |
| Small Dataset Issues | HIGH | Standard ML practice, verified |

**Overall confidence: HIGH**

Core pitfalls (data leakage, random states, p-hacking) are verified with official documentation and recent research. Secondary pitfalls have strong support from multiple credible sources. Some scalability/organization aspects marked LOW due to access issues but are not critical for small-scale experimentation framework.
