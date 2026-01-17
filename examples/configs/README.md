# Experiment Configuration Examples

This directory contains example YAML configuration files for defining experiments.

## Configuration Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `experiment_name` | string | MLflow experiment name (groups related runs) |
| `run_name` | string | Specific run identifier (unique within experiment) |
| `adapter` | string | Which training script adapter to use |
| `parameters` | dict | Hyperparameters passed to training script |
| `tags` | dict | MLflow tags for organization and filtering |
| `random_seed` | int | Random seed for reproducibility |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Human-readable experiment description |
| `sweep` | dict | Parameter sweep definition (see below) |

## Parameter Sweeps

To define a hyperparameter sweep, add a `sweep` section with `grid` or `variables`:

### Grid Search

```yaml
sweep:
  grid:
    lr: [0.0001, 0.0005, 0.001]
    batch_size: [8, 16, 32]
```

This generates all combinations (9 experiments in this case).

### Jinja2 Templating

Use Jinja2 syntax in `run_name` or `parameters`:

```yaml
run_name: "model_lr{{lr}}_bs{{batch_size}}"
parameters:
  learning_rate: {{lr}}
  batch_size: {{batch_size}}
sweep:
  grid:
    lr: [0.0001, 0.001]
    batch_size: [16, 32]
```

## Adapter Types

Adapters wrap existing training scripts for execution via config:

| Adapter | Description | Typical Parameters |
|---------|-------------|-------------------|
| `pytorch` | PyTorch deep learning models | model_name, batch_size, epochs, learning_rate |
| `sklearn` | Scikit-learn models | model_type, alpha, n_estimators |
| `custom` | Custom training scripts | Varies by script |

## Common Parameter Patterns

### PyTorch Image Models

```yaml
parameters:
  model_name: "efficientnet_b0"  # or resnet50, vit_small_patch16_224, etc.
  batch_size: 16
  epochs: 30
  learning_rate: 0.0001
  image_size: 224
  pretrained: true
  use_tta: false
```

### Scikit-learn Models

```yaml
parameters:
  model_type: "ridge"
  alpha: 1.0
  normalize_features: true
```

### Cross-Validation

```yaml
parameters:
  n_folds: 5
  stratified: true
  shuffle: true
```

## Tagging Strategy

Use tags consistently for organization:

```yaml
tags:
  model_type: "cnn"  # cnn, mlp, transformer, sklearn
  purpose: "baseline"  # baseline, hyperparameter_search, ablation, production
  phase: "single_model"  # single_model, ensemble, stacking
  dataset: "canonical_split"  # identify data split used
```

## Examples

- **basic_experiment.yaml** - Simple single experiment configuration
- **sweep_experiment.yaml** - Hyperparameter sweep with grid search
- **adapter_examples/pytorch_effnet.yaml** - PyTorch EfficientNet with parameter sweep
- **adapter_examples/sklearn_ridge.yaml** - Scikit-learn Ridge regression with alpha sweep
- **adapter_examples/xgboost_advanced.yaml** - XGBoost with tree hyperparameter sweep

## Usage

Run experiments using the configuration system:

```python
from mlflow_tracking import ExperimentConfig
import yaml

with open('examples/configs/basic_experiment.yaml') as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig(**config_dict)
# Use config with adapter to run experiment
```

## Best Practices

1. **Descriptive names**: Use clear experiment_name and run_name values
2. **Consistent tags**: Use standard tag values for filtering
3. **Reproducibility**: Always set random_seed explicitly
4. **Documentation**: Add description field for context
5. **Version control**: Commit configs alongside code changes

## Creating Adapters for Additional Scripts

The project has 29 training scripts. Two adapters (PyTorch, Sklearn) demonstrate the pattern.
To wrap the remaining 27 scripts, follow this 3-step pattern:

**Step 1: Create adapter class**

```python
from mlflow_tracking.adapters import BaseAdapter, AdapterRegistry
from mlflow_tracking import ExperimentConfig
import subprocess
import json
from typing import Dict

@AdapterRegistry.register('pytorch_resnet')
class ResNetAdapter(BaseAdapter):
    def validate_config(self, config: ExperimentConfig) -> bool:
        # Define required parameters for this script
        required = ['model_name', 'batch_size', 'epochs']
        for param in required:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        return True

    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        script_path = "scripts/train_resnet.py"  # Update script path
        args = ["python3", script_path]

        # Build CLI args from config.parameters
        for param_name, value in config.parameters.items():
            args.extend([f"--{param_name}", str(value)])

        result = subprocess.run(args, capture_output=True, text=True, check=True)
        metrics_dict = json.loads(result.stdout.split('\n')[-1])  # Parse JSON output

        # Convert to MLflow format
        return {k.replace('_', '.'): float(v) for k, v in metrics_dict.items()}
```

**Step 2: Create YAML config**

```yaml
experiment_name: resnet_experiments
run_name: resnet_b0
adapter: pytorch_resnet  # Use new adapter name
parameters:
  model_name: "resnet18"
  batch_size: 32
  epochs: 30
```

**Step 3: Run experiment**

```bash
exp-run configs/resnet_config.yaml
```

That's the entire pattern. Same 3 steps for all remaining scripts.

### Adapter Implementation Notes

**Required methods:**
- `validate_config(config)`: Check required parameters exist
- `execute(config, tracker)`: Run script and return metrics dict

**Script output format:**
Scripts must print JSON as the last line of stdout:
```python
print(json.dumps({"train_rmse": 8.2, "val_rmse": 10.5}))
```

**Metric naming:**
Adapters convert underscore keys to dot notation for MLflow:
- `train_rmse` -> `train.rmse`
- `val_r2` -> `val.r2`

**Error handling:**
- Raise `ValueError` for validation failures
- Let `subprocess.CalledProcessError` propagate for script failures
- Adapter returns metrics; CLI logs them to MLflow

### Available Adapters

Currently implemented adapters:

| Adapter | Script | Required Parameters |
|---------|--------|---------------------|
| `pytorch` | `train_oof_effnet.py` | model_name, batch_size, epochs, learning_rate |
| `sklearn` | `train_ridge_advanced.py` | model_type, random_seed |

To add more adapters, copy the pattern from `mlflow_tracking/adapters.py`.
