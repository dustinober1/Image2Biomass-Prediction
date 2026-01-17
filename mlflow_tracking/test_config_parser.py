"""
Test script demonstrating configuration parsing features.

This script shows how to use ConfigParser to:
1. Load basic experiment configurations from YAML
2. Expand parameter sweeps into multiple configurations
3. Validate configurations against adapter requirements
4. Use Jinja2 templating for variable substitution
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Configuration Parser Demo")
print("=" * 60)

# ============================================================================
# 1. Basic Config Loading
# ============================================================================
print("\n--- 1. Basic Config Loading ---\n")

# Temporarily import from config_parser module until exports are updated
from mlflow_tracking.config_parser import ConfigParser, ExperimentConfig

# Load basic experiment
config = ConfigParser.load_config('examples/configs/basic_experiment.yaml')
print(f"Experiment: {config.experiment_name}")
print(f"Run: {config.run_name}")
print(f"Adapter: {config.adapter}")
print(f"Description: {config.description}")
print(f"Parameters: {config.parameters}")
print(f"Tags: {config.tags}")
print(f"Random Seed: {config.random_seed}")

# ============================================================================
# 2. Sweep Expansion
# ============================================================================
print("\n--- 2. Sweep Expansion ---\n")

# Expand parameter sweep
configs = ConfigParser.expand_sweeps('examples/configs/sweep_experiment.yaml')
print(f"Sweep expanded into {len(configs)} configurations:")
for i, cfg in enumerate(configs):
    print(f"  {i+1}. {cfg.run_name}")
    print(f"     lr={cfg.parameters['learning_rate']}, bs={cfg.parameters['batch_size']}")

# ============================================================================
# 3. Template Substitution
# ============================================================================
print("\n--- 3. Template Substitution ---\n")

from jinja2 import Template

# Show how Jinja2 templating works
print("Jinja2 template rendering examples:")
template_str = "run_{{lr}}_bs{{bs}}"
template = Template(template_str)

test_values = [
    (0.001, 8),
    (0.001, 16),
    (0.01, 8),
    (0.01, 16),
]

for lr, bs in test_values:
    rendered = template.render(lr=lr, bs=bs)
    print(f"  {template_str} with lr={lr}, bs={bs} -> {rendered}")

# ============================================================================
# 4. Validation Errors
# ============================================================================
print("\n--- 4. Validation Tests ---\n")

# Test 4a: Missing required field
print("Test 4a: Missing required field")
try:
    invalid_yaml = """
    experiment_name: test
    # Missing run_name, adapter, parameters
    """
    import yaml
    from io import StringIO
    config_dict = yaml.safe_load(StringIO(invalid_yaml))
    from mlflow_tracking.config_parser import ExperimentConfig
    config = ExperimentConfig(**config_dict)
    print("  ERROR: Should have raised validation error")
except Exception as e:
    print(f"  Caught expected error: {type(e).__name__}")
    print(f"  Message: {e}")

# Test 4b: Invalid adapter (requires AdapterRegistry)
print("\nTest 4b: Unknown adapter validation")
try:
    # Create a basic config with unknown adapter
    from mlflow_tracking.adapters import AdapterRegistry

    test_config = ExperimentConfig(
        experiment_name="test",
        run_name="test_run",
        adapter="unknown_adapter",
        parameters={"lr": 0.001},
        tags={}
    )

    ConfigParser.validate(test_config, AdapterRegistry)
    print("  ERROR: Should have raised ValueError for unknown adapter")
except ValueError as e:
    print(f"  Caught expected error: ValueError")
    print(f"  Message: {e}")

# ============================================================================
# 5. Using ConfigParser in Your Own Scripts
# ============================================================================
print("\n--- 5. Usage Guide ---\n")

usage_guide = """
How to use ConfigParser in your own scripts:

1. Load a single config:
   config = ConfigParser.load_config('path/to/config.yaml')

2. Expand a parameter sweep:
   configs = ConfigParser.expand_sweeps('path/to/sweep_config.yaml')

3. Validate against adapter registry:
   ConfigParser.validate(config, AdapterRegistry)

4. Access configuration fields:
   experiment_name = config.experiment_name
   parameters = config.parameters
   tags = config.tags

5. Use with ExperimentTracker:
   from mlflow_tracking import ExperimentTracker
   tracker = ExperimentTracker(config.experiment_name)
   with tracker.start_run(config.run_name):
       tracker.log_params(config.parameters)
       tracker.log_tags(config.tags)

Differences between load_config() and expand_sweeps():
- load_config(): Loads a single config from YAML
- expand_sweeps(): Loads config and expands sweep.grid into multiple configs

How sweep.grid works:
- Defines a grid search over parameter values
- Generates all combinations using itertools.product
- Jinja2 templates in run_name and parameters are substituted
- Example: {'lr': [0.001, 0.01], 'bs': [8, 16]} -> 4 configs

Jinja2 syntax:
- Variables: {{variable_name}}
- Can be used in run_name and parameters
- Example: run_name: "model_{{lr}}_bs{{bs}}"
- Template is rendered with sweep values
"""

print(usage_guide)

# ============================================================================
# 6. Advanced: Non-sweep config handling
# ============================================================================
print("\n--- 6. Non-Sweep Config Handling ---\n")

# expand_sweeps() also works on configs without sweeps
single_configs = ConfigParser.expand_sweeps('examples/configs/basic_experiment.yaml')
print(f"Non-sweep config expanded into {len(single_configs)} config(s)")
print(f"Config: {single_configs[0].run_name}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
ConfigParser Features Demonstrated:
1. Basic config loading - Load and validate YAML configs
2. Sweep expansion - Generate multiple configs from grid definition
3. Template substitution - Jinja2 rendering of {{variables}}
4. Validation - Catch missing fields and invalid adapters
5. Adapter integration - Validate configs against adapter requirements

Key Points:
- Use yaml.safe_load() for security
- Use Pydantic for schema validation
- Use Jinja2 Template for variable substitution
- Use itertools.product() for grid search expansion
- ConfigParser.validate() checks adapter existence and config validity
""")
