# Experiment Tracking Framework

A systematic MLflow-based framework for tracking machine learning experiments with automatic metadata logging, canonical data splits, and reproducibility guarantees.

## Overview

This framework provides a Python SDK for tracking ML experiments that addresses common pitfalls in machine learning research:

- **Automatic experiment logging** with timestamps, status tracking (running/completed/failed), and duration
- **Canonical data splits** to prevent data leakage and enable fair model comparison
- **Environment tracking** (git commits, package versions, system info) for reproducibility
- **No cherry-picking** - all experiments are logged, including failures
- **Context manager support** for automatic error handling

## Installation

Install dependencies:

```bash
pip install mlflow numpy pandas scikit-learn
```

The framework uses MLflow with a local SQLite backend by default - no additional setup required.

## Quick Start

```python
from mlflow_tracking import ExperimentTracker, DataSplitter
import numpy as np

# Create experiment tracker
with ExperimentTracker("my_experiment", auto_log_environment=True) as tracker:
    # Start a run
    run_id = tracker.start_run("experiment_1", random_seed=42)

    # Log hyperparameters
    tracker.log_params({"learning_rate": 0.01, "n_estimators": 100})

    # Load canonical splits
    splitter = DataSplitter()
    train_idx, val_idx, test_idx = splitter.get_split_indices()

    # ... train model ...

    # Log metrics
    tracker.log_metrics({"rmse": 10.5, "r2": 0.85})
```

View results:

```bash
mlflow ui
# Open http://localhost:5000
```

## Detailed Usage

### ExperimentTracker

The main interface for tracking experiments.

#### Initialization

```python
from mlflow_tracking import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="my_experiment",
    auto_log_environment=True  # Auto-log git hash, package versions
)
```

#### Methods

**start_run()** - Start a new experiment run

```python
run_id = tracker.start_run(
    run_name="baseline_rf",
    tags={"purpose": "baseline"},
    random_seed=42
)
```

**log_params()** - Log hyperparameters or configuration

```python
tracker.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5
})
```

**log_metrics()** - Log evaluation metrics

```python
tracker.log_metrics({
    "train.rmse": 8.2,
    "val.rmse": 10.5,
    "test.rmse": 11.3,
    "train.r2": 0.90,
    "val.r2": 0.85,
    "test.r2": 0.82
})
```

**log_artifact()** - Save model files or outputs

```python
import joblib

joblib.dump(model, "models/model.pkl")
tracker.log_artifact("models/model.pkl")
```

#### Context Manager (Recommended)

Use the context manager for automatic error handling:

```python
with ExperimentTracker("my_experiment") as tracker:
    tracker.start_run("experiment_1")

    # Your training code here
    # ...

    # If an exception occurs, run is automatically marked as "failed"
    # Otherwise, marked as "completed"
```

### DataSplitter

Manage canonical train/validation/test splits to prevent data leakage.

#### Creating Canonical Splits

```python
from mlflow_tracking import DataSplitter
import numpy as np

# Load your data
X = np.array(...)  # Features
y = np.array(...)  # Targets

# Create and save splits
splitter = DataSplitter(split_file="data/canonical_splits.json")

# Option 1: Simple split
splits = splitter.create_splits(X, y)
splitter.save_splits()

# Option 2: Stratified split (recommended for imbalanced data)
# Bin targets into quantiles for stratification
n_bins = 5
y_binned = np.floor(n_bins * (y - y.min()) / (y.max() - y.min())).astype(int)
splits = splitter.create_splits(X, y, stratify=y_binned)
splitter.save_splits()
```

#### Using Canonical Splits

```python
# Load existing splits
splitter = DataSplitter()
train_idx, val_idx, test_idx = splitter.get_split_indices()

# Apply splits
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# Validate splits (check no overlap, distribution maintained)
stats = splitter.validate_splits(X, y)
print(f"Train: {stats['n_train']}, Val: {stats['n_val']}, Test: {stats['n_test']}")
print(f"Target means - Train: {stats['train_mean']:.2f}, "
      f"Val: {stats['val_mean']:.2f}, Test: {stats['test_mean']:.2f}")
```

#### Convenience Function

```python
from mlflow_tracking import create_canonical_splits

# Create and save in one call
splits = create_canonical_splits(X, y, split_file="data/canonical_splits.json")
```

### Environment Tracking

Environment metadata is automatically logged when `auto_log_environment=True`.

#### What Gets Logged

**Tags:**
- `git.commit_hash`: Current git commit SHA
- `git.branch`: Current git branch
- `system.os`: Operating system
- `system.architecture`: System architecture (e.g., x86_64)
- `python.version`: Python version

**Parameters (prefixed with `env.`):**
- `env.python`: Python version
- `env.numpy`: NumPy version
- `env.pandas`: Pandas version
- `env.scikit-learn`: scikit-learn version
- `env.torch`: PyTorch version (if installed)
- `env.xgboost`: XGBoost version (if installed)
- `env.mlflow`: MLflow version
- `env.shap`: SHAP version (if installed)

#### Manual Environment Logging

```python
from mlflow_tracking import get_environment

env = get_environment(packages=["numpy", "scikit-learn", "torch"])
tracker.log_environment(env)
```

### Organization and Discovery

Organize experiments into logical groups and discover insights through powerful search capabilities.

#### ExperimentOrganizer

The `ExperimentOrganizer` class provides methods for grouping, tagging, and searching experiments.

**Initialization:**
```python
from mlflow_tracking import ExperimentOrganizer

organizer = ExperimentOrganizer()
```

**Create experiment groups:**
```python
# Create a group for ablation studies
exp_id = organizer.create_group("ablation-studies", tags={
    "purpose": "feature_ablation",
    "project": "biomass"
})

# Create a group for ensemble tests
exp_id = organizer.create_group("ensemble-tests", tags={
    "purpose": "ensemble_methods"
})
```

**List all groups:**
```python
groups = organizer.list_groups()
for group in groups:
    print(f"{group['name']}: {group.get('tags', {})}")
```

**Search experiments:**
```python
# Search by metrics
runs = organizer.search_runs(filter_string="metrics.val_rmse < 10.0")

# Search by parameters
runs = organizer.search_runs(filter_string="params.model_type = 'random_forest'")

# Search by tags
runs = organizer.search_runs(filter_string="tags.status = 'completed'")

# Combined search
runs = organizer.search_runs(
    filter_string="params.model_type = 'xgboost' and metrics.test_rmse < 15.0"
)

# Order results
runs = organizer.search_runs(
    filter_string="metrics.val_rmse < 12.0",
    order_by=["metrics.val_rmse ASC"]
)

# Limit results
runs = organizer.search_runs(max_results=10)
```

**Find best runs:**
```python
# Get top 5 runs by validation RMSE
best_runs = organizer.get_best_runs(
    group_name="ablation-studies",
    metric_name="val_rmse",
    top_k=5
)

for i, run in enumerate(best_runs, 1):
    print(f"#{i}: Run {run['run_id']}")
    print(f"   Val RMSE: {run['metrics']['val_rmse']:.2f}")
    print(f"   Model: {run['tags'].get('model_type')}")
```

**Add tags to existing runs:**
```python
organizer.add_tags_to_run(run_id, {
    "model_type": "random_forest",
    "purpose": "baseline",
    "status": "completed"
})
```

#### ExperimentTracker Organization Methods

The `ExperimentTracker` class includes convenience methods for organizing runs.

**Set experiment group:**
```python
tracker = ExperimentTracker("my_experiment")

# Set group BEFORE starting run
tracker.set_group("ablation-studies")
tracker.start_run("experiment_1")

# All subsequent runs in this group
tracker.start_run("experiment_2")  # Also in "ablation-studies"
```

**Add tags to active run:**
```python
tracker = ExperimentTracker("my_experiment")
tracker.start_run("experiment_1")

# Add tags for filtering and organization
tracker.add_tags({
    "model_type": "xgboost",
    "purpose": "hyperparameter_tuning",
    "phase": "development"
})
```

**Get run ID:**
```python
tracker.start_run("experiment_1")
run_id = tracker.get_run_id()
print(f"Current run: {run_id}")

# Use with ExperimentOrganizer
organizer.add_tags_to_run(run_id, {"extra_tag": "value"})
```

#### MLflow Filter Syntax Reference

MLflow supports powerful filter strings for searching experiments:

**Metrics:**
```
metrics.val_rmse < 10.0
metrics.test_r2 > 0.8
metrics.train_loss <= 0.5
```

**Parameters:**
```
params.model_type = 'random_forest'
params.learning_rate > 0.01
params.n_estimators >= 100
```

**Tags:**
```
tags.status = 'completed'
tags.model_type = 'xgboost'
```

**Combined (AND only - OR not supported):**
```
params.model_type = 'xgboost' and metrics.val_rmse < 10.0
tags.status = 'completed' and metrics.test_rmse < 15.0
params.n_estimators >= 100 and params.max_depth <= 20
```

**Ordering:**
```
order_by=["metrics.val_rmse ASC"]      # Best first
order_by=["metrics.test_r2 DESC"]      # Highest first
order_by=["params.n_estimators ASC"]   # Sort by param
```

#### Web UI Usage

The MLflow UI provides a visual interface for exploring experiments.

**Start the UI:**
```bash
# From project root
mlflow ui

# UI opens at http://localhost:5000
```

**Navigate experiments:**
- **Experiments sidebar**: View all experiment groups
- **Runs table**: See runs within each group with metrics and params
- **Compare runs**: Select multiple runs to compare side-by-side
- **Artifacts**: Download models, predictions, and other outputs

**Search in UI:**
```
# In the search box, use filter syntax
metrics.val_rmse < 10
params.max_depth = 10
tags.model_type = "xgboost"
```

**View experiment details:**
- Click on a run to see detailed metrics, parameters, and tags
- View artifacts (saved models, plots, CSVs)
- Check git commit and package versions for reproducibility

## Requirements Coverage

This framework addresses the following requirements from Phase 1 and Phase 2:

### Tracking (TRACK-01 through TRACK-05)

- **TRACK-01**: Records each experiment with timestamp, status, and duration
  - Implemented: `start_run()` sets start_time, `end_run()` sets end_time and status
- **TRACK-02**: Captures all hyperparameters and configuration values
  - Implemented: `log_params()` stores arbitrary parameters
- **TRACK-03**: Records evaluation metrics (RMSE, R², MAE)
  - Implemented: `log_metrics()` stores numeric metrics
- **TRACK-04**: Stores artifacts (model files, predictions, analysis outputs)
  - Implemented: `log_artifact()` saves files to MLflow artifact store
- **TRACK-05**: Provides Python SDK for programmatic logging
  - Implemented: All functionality exposed via Python API

### Reproducibility (REPRO-01 through REPRO-03)

- **REPRO-01**: Tracks Python environment (package versions)
  - Implemented: `get_environment()` captures package versions, git hash, system info
- **REPRO-02**: Enforces proper data splitting (train/validation/test)
  - Implemented: `DataSplitter` creates canonical splits with validation
- **REPRO-03**: Logs ALL experiments including failures (prevents cherry-picking)
  - Implemented: Context manager marks failed runs, framework doesn't filter results

### Organization (ORG-01 through ORG-04)

- **ORG-01**: Group experiments into logical collections
  - Implemented: `ExperimentOrganizer.create_group()` creates MLflow experiments
  - Implemented: `ExperimentTracker.set_group()` sets active experiment for runs
- **ORG-02**: Tag experiments for filtering and organization
  - Implemented: `ExperimentTracker.add_tags()` adds tags to active runs
  - Implemented: `ExperimentOrganizer.add_tags_to_run()` adds tags post-hoc
- **ORG-03**: Search experiments by metrics, parameters, and tags
  - Implemented: `ExperimentOrganizer.search_runs()` with MLflow filter syntax
  - Implemented: Support for metric, parameter, and tag filtering
  - Implemented: Result ordering and limiting
- **ORG-04**: Web UI for experiment exploration
  - Implemented: MLflow built-in UI at `http://localhost:5000`
  - Implemented: Visual comparison of runs, artifacts, and metadata
  - Implemented: Search and filter in UI

## Best Practices

Based on research documented in `.planning/research/PITFALLS.md`:

### 1. Always Use Context Manager

```python
# GOOD - automatic error tracking
with ExperimentTracker("experiment") as tracker:
    tracker.start_run("run_1")
    train_model()
    # Exceptions automatically mark run as "failed"

# BAD - manual error handling prone to mistakes
tracker = ExperimentTracker("experiment")
tracker.start_run("run_1")
try:
    train_model()
    tracker.end_run(status="completed")
except:
    tracker.end_run(status="failed")  # Easy to forget
```

### 2. Use Canonical Splits for Fair Comparison

```python
# GOOD - same splits across experiments
splitter = DataSplitter()
train_idx, val_idx, test_idx = splitter.get_split_indices()

# BAD - different splits each time
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None  # Different each run!
)
```

### 3. Log Metrics for All Splits

```python
# GOOD - report train/val/test separately
tracker.log_metrics({
    "train.rmse": 8.2,
    "val.rmse": 10.5,
    "test.rmse": 11.3
})

# BAD - only test metric (can't detect overfitting)
tracker.log_metrics({"rmse": 11.3})
```

### 4. Never Tune on Test Set

```python
# GOOD - tune on validation, report test once
for lr in [0.001, 0.01, 0.1]:
    score = train_and_evaluate(X_train, y_train, X_val, y_val, lr=lr)
    tracker.log_metrics({"val.score": score})

# Final evaluation ONCE
final_model = train(X_train, y_train, X_val, y_val, best_lr=0.01)
test_score = evaluate(final_model, X_test, y_test)
tracker.log_metrics({"test.score": test_score})

# BAD - tuning on test set (p-hacking)
for lr in [0.001, 0.01, 0.1]:
    score = train_and_evaluate(X_train, y_train, X_test, y_test, lr=lr)
    # This is p-hacking!
```

### 5. Use Pipelines to Prevent Data Leakage

```python
# GOOD - preprocessing in pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

model = make_pipeline(StandardScaler(), RandomForestRegressor())
model.fit(X_train, y_train)  # Scaler fit on training data only

# BAD - global preprocessing causes leakage
scaler = StandardScaler().fit(X)  # Leaks test info!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 6. Log Random Seed

```python
# GOOD - seed logged for reproducibility
tracker.start_run("experiment_1", random_seed=42)

# Also set seed in your code
np.random.seed(42)

# BAD - no seed, results irreproducible
tracker.start_run("experiment_1")
```

### 7. Report All Results (No Cherry-Picking)

```python
# GOOD - log all experiments
for max_depth in [5, 10, 15, 20]:
    with ExperimentTracker("tuning") as tracker:
        run_id = tracker.start_run(f"depth_{max_depth}")
        score = train_model(max_depth=max_depth)
        tracker.log_metrics({"test.rmse": score})
    # All experiments logged, not just best

# BAD - only report best
results = []
for max_depth in [5, 10, 15, 20]:
    results.append(train_model(max_depth=max_depth))
best_score = min(results)  # Only reporting winner = cherry-picking
```

## Examples

### Basic Usage

```python
from mlflow_tracking import ExperimentTracker

with ExperimentTracker("baseline_experiment") as tracker:
    run_id = tracker.start_run("random_forest_v1")

    tracker.log_params({"n_estimators": 100, "max_depth": 10})
    tracker.log_metrics({"rmse": 12.3, "r2": 0.78})
```

### With Canonical Splits

```python
from mlflow_tracking import ExperimentTracker, DataSplitter

splitter = DataSplitter()
train_idx, val_idx, test_idx = splitter.get_split_indices()

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

with ExperimentTracker("with_canonical_splits") as tracker:
    tracker.start_run("rf_baseline")
    model = train_model(X_train, y_train)

    val_metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)

    tracker.log_metrics({f"val.{k}": v for k, v in val_metrics.items()})
    tracker.log_metrics({f"test.{k}": v for k, v in test_metrics.items()})
```

### Hyperparameter Tuning (Multiple Experiments)

```python
from mlflow_tracking import ExperimentTracker, DataSplitter

splitter = DataSplitter()
train_idx, val_idx, test_idx = splitter.get_split_indices()

results = []
for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, 15]:
        with ExperimentTracker("hyperparameter_tuning") as tracker:
            run_name = f"rf_est_{n_estimators}_depth_{max_depth}"
            tracker.start_run(run_name)

            tracker.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth
            })

            model = train_model(
                X[train_idx], y[train_idx],
                n_estimators=n_estimators,
                max_depth=max_depth
            )

            val_metrics = evaluate(model, X[val_idx], y[val_idx])
            test_metrics = evaluate(model, X[test_idx], y[test_idx])

            tracker.log_metrics({f"val.{k}": v for k, v in val_metrics.items()})
            tracker.log_metrics({f"test.{k}": v for k, v in test_metrics.items()})

            results.append({
                "run_id": tracker.active_run.info.run_id,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                **test_metrics
            })

# All experiments logged - find best in MLflow UI
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.sort_values("rmse").head())
```

### Error Handling

```python
from mlflow_tracking import ExperimentTracker

try:
    with ExperimentTracker("error_demo") as tracker:
        tracker.start_run("will_fail")
        tracker.log_params({"invalid_param": 0})

        # This will raise an exception
        model = train_model_with_invalid_params()
except Exception as e:
    print(f"Caught error: {e}")
    # Run is automatically marked as "failed" in MLflow
```

## MLflow UI Commands

### Start MLflow UI

```bash
# From project root
mlflow ui

# UI opens at http://localhost:5000
```

### Navigate Experiments

- **Experiments**: View all experiments and runs
- **Compare**: Select multiple runs to compare params and metrics
- **Artifacts**: Download model files and outputs

### Search and Filter

```python
# In MLflow UI, use query syntax:
# metrics.test_rmse < 10
# params.max_depth = 10
# tags.status = "completed"
```

### Export Results

```bash
# Export experiment runs to CSV
mlflow experiments csv -i <experiment_id> -o results.csv
```

## Troubleshooting

### Issue: "No active run" error

**Cause**: Calling `log_params()` or `log_metrics()` before `start_run()`

**Solution**:
```python
with ExperimentTracker("exp") as tracker:
    tracker.start_run("run_1")  # Call start_run first
    tracker.log_params({...})    # Then log
```

### Issue: Split file not found

**Cause**: Calling `get_split_indices()` before creating splits

**Solution**:
```python
# First create splits (one time)
splitter = DataSplitter()
splitter.create_splits(X, y, stratify=y_binned)
splitter.save_splits()

# Then load in subsequent runs
splitter = DataSplitter()
train_idx, val_idx, test_idx = splitter.get_split_indices()
```

### Issue: Git hash shows "unknown"

**Cause**: Running outside a git repository or git not installed

**Solution**: This is expected behavior - framework gracefully handles missing git

### Issue: Package versions show "not_installed"

**Cause**: Optional packages not in environment

**Solution**: Either install the package or ignore - framework logs what's available

### Issue: MLflow UI shows old experiments

**Cause**: MLflow tracking URI pointing to different database

**Solution**: Check `mlflow_tracking/config.py` tracking URI matches your project

## Full Example

See `mlflow_tracking/full_example.py` for a complete working example demonstrating:

- Data loading and canonical splits
- Experiment tracking with context manager
- Parameter and metric logging
- Model artifact storage
- Error handling
- Multiple experiments (no cherry-picking)

See `mlflow_tracking/test_organization.py` for organization and discovery examples:

- Creating experiment groups
- Tagging experiments for filtering
- Searching by metrics, parameters, and tags
- Finding best runs
- Using MLflow UI for exploration

Run the examples:

```bash
python mlflow_tracking/full_example.py
python mlflow_tracking/test_organization.py
```

## Architecture

```
mlflow_tracking/
├── __init__.py           # Package exports
├── tracker.py            # ExperimentTracker class (with tagging methods)
├── organizer.py          # ExperimentOrganizer class (grouping & search)
├── data_split.py         # DataSplitter class
├── environment.py        # Environment capture functions
├── config.py             # MLflow configuration
├── full_example.py       # Complete working example
├── test_organization.py  # Organization features demo
├── test_splits.py        # Data split validation tests
└── README.md             # This file
```

## Data Storage

- **MLflow database**: `mlflow_tracking/mlruns.db` (SQLite)
- **MLflow artifacts**: `mlflow_tracking/mlruns/` (run outputs)
- **Canonical splits**: `data/canonical_splits.json` (train/val/test indices)

## References

- MLflow documentation: https://mlflow.org/docs/latest/index.html
- Project requirements: `.planning/REQUIREMENTS.md`
- Research on pitfalls: `.planning/research/PITFALLS.md`
- Phase 1 plan: `.planning/phases/01-experiment-tracking-foundation/01-04-PLAN.md`
- Phase 2 plan: `.planning/phases/02-organization-discovery/02-01-PLAN.md`

---

**Version**: 0.2.0
**Last updated**: 2026-01-17
**Phase**: 2 - Organization & Discovery
