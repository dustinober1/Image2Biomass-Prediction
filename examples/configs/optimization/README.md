# Hyperparameter Optimization with Optuna

This directory contains example configurations for automated hyperparameter optimization using Optuna. Optuna searches hyperparameter spaces efficiently, pruning underperforming trials early to save computation time.

## Introduction

Hyperparameter optimization is the process of finding the best hyperparameter values for a machine learning model. Instead of manually trying different values, Optuna automates this search using:

- **Bayesian optimization**: Learns from previous trials to suggest promising hyperparameters
- **Pruning**: Stops underperforming trials early to save computation
- **Parallel trials**: Runs multiple trials simultaneously (with `--n-jobs`)
- **MLflow integration**: Logs all trials to MLflow for analysis

## Search Space Syntax

Hyperparameter search spaces are defined in the `optimization.search` section. Each parameter has a type and corresponding configuration:

### Float Parameters

For continuous values like learning rate, regularization strength, etc.

```yaml
search:
  learning_rate:
    type: float
    low: 0.00001    # Lower bound
    high: 0.1       # Upper bound
    log: true       # Use log-scale sampling (recommended for LR, alpha)
```

### Integer Parameters

For discrete values like max depth, number of estimators, batch size.

```yaml
search:
  max_depth:
    type: int
    low: 3          # Lower bound (inclusive)
    high: 10        # Upper bound (inclusive)
    step: 1         # Step size between values

  n_estimators:
    type: int
    low: 50
    high: 500
    step: 50        # Will try 50, 100, 150, ..., 500
```

### Categorical Parameters

For discrete choices like optimizer type, activation function, etc.

```yaml
search:
  optimizer:
    type: categorical
    choices:
      - adam
      - sgd
      - adamw

  activation:
    type: categorical
    choices:
      - relu
      - gelu
      - swish
```

## Pruner Types

Pruners stop unpromising trials early to save computation. Three pruner types are supported:

### Median Pruner (Recommended for starting)

Stops trials if their intermediate metrics are worse than the median of previous trials.

```yaml
pruner:
  type: median
  n_startup_trials: 5    # Don't prune until 5 trials complete
  n_warmup_steps: 10     # Don't prune until step 10 in each trial
  interval_steps: 1      # Check pruning every step
```

### Hyperband Pruner (Aggressive pruning)

Uses the Hyperband algorithm for aggressive early stopping. Best for large search spaces.

```yaml
pruner:
  type: hyperband
  min_resource: 1        # Minimum resource (e.g., epochs)
  max_resource: 100      # Maximum resource
  reduction_factor: 3    # How aggressively to prune
```

### Successive Halving Pruner

Similar to Hyperband but with simpler configuration.

```yaml
pruner:
  type: successive_halving
  reduction_factor: 4        # Stop worst 75% at each iteration
  min_early_stopping_rate: 0
```

## CLI Usage

### Basic Usage

Run optimization with default settings:

```bash
exp-run-optimize examples/configs/optimization/01_effnet_lr_search.yaml
```

### Override Trial Count

Run a quick test with fewer trials:

```bash
exp-run-optimize 01_effnet_lr_search.yaml --n-trials 10
```

### Parallel Trials

Run multiple trials in parallel (requires adequate resources):

```bash
exp-run-optimize 01_effnet_lr_search.yaml --n-jobs 4
```

### Auto-detect Parallel Jobs

Let the framework automatically determine safe parallelism:

```bash
exp-run-optimize 01_effnet_lr_search.yaml --n-jobs -1
```

### Verbose Output

See detailed optimization progress:

```bash
exp-run-optimize 01_effnet_lr_search.yaml --verbose
```

### Combined Options

```bash
exp-run-optimize 01_effnet_lr_search.yaml --n-trials 50 --n-jobs 4 --verbose
```

## Optimization Config Schema

The `optimization` section in your YAML config must have:

```yaml
optimization:
  n_trials: 100              # Number of trials to run
  study_name: my_study       # Study name for persistence
  direction: minimize        # "minimize" or "maximize"
  metric: val.rmse           # Metric to optimize
  search:                    # Search space (required)
    param_name:
      type: float|int|categorical
      # ... type-specific fields
  pruner:                    # Optional pruner config
    type: median|hyperband|successive_halving
    # ... pruner-specific fields
  timeout: 3600              # Optional: max study duration in seconds
```

## Best Practices

### 1. Start with Coarse Search

Begin with a broad search space to identify promising regions:

```yaml
# Good for initial exploration
learning_rate:
  type: float
  low: 0.00001
  high: 0.1
  log: true
```

### 2. Refine with Fine Search

Once you've identified a promising region, refine the search:

```yaml
# Good for refinement (after coarse search finds LR ~ 0.001)
learning_rate:
  type: float
  low: 0.0005
  high: 0.005
  log: true
```

### 3. Use Log-Scale for Ratios

Always use `log: true` for learning rate, alpha, and other ratio-based parameters:

```yaml
learning_rate:
  type: float
  low: 1e-5
  high: 1e-1
  log: true  # Critical for LR search!

alpha:
  type: float
  low: 0.001
  high: 100.0
  log: true  # Critical for regularization!
```

### 4. Choose Right Pruner

- **Median pruner**: Good starting point, conservative
- **Hyperband**: Best for large search spaces, aggressive pruning
- **Successive halving**: Simpler alternative to Hyperband

### 5. Set Appropriate n_warmup_steps

Don't prune too early - let trials show their potential:

```yaml
pruner:
  type: median
  n_startup_trials: 5     # Build history before pruning
  n_warmup_steps: 10      # Let each trial run for 10 steps
```

### 6. Use Parallel Trials Cautiously

Parallel trials speed up optimization but don't share information:

```bash
# Good: Moderate parallelism for exploration
exp-run-optimize config.yaml --n-jobs 4

# Risky: High parallelism wastes trials (no info sharing)
exp-run-optimize config.yaml --n-jobs 16
```

## Interpreting Results

After optimization completes, you'll see:

```
============================================================
Optimization complete!
============================================================
Best value: 9.876
Best params:
  learning_rate: 0.0032
  batch_size: 32

Best config saved to: examples/configs/optimization/01_effnet_lr_search_best.yaml

To run with best hyperparameters:
  exp-run examples/configs/optimization/01_effnet_lr_search_best.yaml
```

### Key Metrics

- **Best value**: The best metric value found (e.g., lowest RMSE)
- **Best params**: Hyperparameter values that achieved the best value
- **Best config YAML**: Ready-to-run config with optimal hyperparameters

### Running with Best Hyperparameters

The optimization automatically saves a `_best.yaml` config:

```bash
# Run final experiment with optimal hyperparameters
exp-run examples/configs/optimization/01_effnet_lr_search_best.yaml
```

## Example Configurations

### 01_effnet_lr_search.yaml

Learning rate optimization for EfficientNet-B0 using log-scale sampling.

**Key features:**
- Single parameter (learning_rate)
- Log-scale sampling (critical for LR)
- Median pruner with conservative settings

**Usage:**
```bash
exp-run-optimize 01_effnet_lr_search.yaml --n-trials 100 --verbose
```

### 02_ridge_alpha_search.yaml

Alpha (regularization strength) optimization for Ridge regression.

**Key features:**
- Single parameter (alpha)
- Log-scale sampling (critical for regularization)
- Median pruner

**Usage:**
```bash
exp-run-optimize 02_ridge_alpha_search.yaml --n-trials 50
```

### 03_xgboost_multi_param.yaml

Multi-parameter optimization for XGBoost (5 parameters).

**Key features:**
- Multiple parameters (learning_rate, max_depth, n_estimators, subsample, colsample_bytree)
- Mix of float (log and linear scale) and int parameters
- Hyperband pruner for aggressive early stopping

**Usage:**
```bash
exp-run-optimize 03_xgboost_multi_param.yaml --n-trials 100 --n-jobs 4
```

## Analyzing Results in MLflow

All optimization trials are logged to MLflow for analysis:

```bash
# Start MLflow UI
mlflow ui

# Navigate to http://localhost:5000
# View your experiment and compare trials
```

### Key MLflow Features

- **Trial comparison**: Compare metrics across all trials
- **Parameter importance**: See which hyperparameters matter most
- **Visualization**: Plot optimization history and parameter relationships

## Troubleshooting

### Error: "Config file must have 'optimization' section"

Make sure your YAML has an `optimization` section:

```yaml
experiment_name: my_experiment
run_name: my_run
adapter: pytorch
parameters: {...}
optimization:          # Required!
  n_trials: 100
  study_name: my_study
  # ...
```

### Error: "Metric 'val.rmse' not found in trial results"

Your training script must log the metric you're optimizing:

```python
# In your training script
mlflow.log_metrics({"val.rmse": 10.5, "train.rmse": 8.2})
```

### All Trials Failing

Check that:
1. Your base config (parameters) works with `exp-run config.yaml`
2. The metric you're optimizing is actually logged
3. Search space bounds are reasonable (not too wide/narrow)

### Poor Optimization Results

- **Increase n_trials**: More trials = better search
- **Widen search space**: Bounds may be too narrow
- **Use log-scale**: For learning rate, alpha, etc.
- **Adjust pruner**: Less aggressive pruning (higher n_warmup_steps)

## Additional Resources

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorials](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
