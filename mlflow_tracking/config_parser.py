"""
Experiment configuration parser using Pydantic for schema validation.

This module defines the YAML schema for experiment configurations and provides
validation logic to ensure configs are correct before execution.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from jinja2 import Template, TemplateError
import itertools


class ExperimentConfig(BaseModel):
    """
    Schema for experiment configuration defined in YAML.

    This class defines what a valid experiment configuration looks like,
    including required fields, optional fields, and validation rules.

    Attributes:
        experiment_name: MLflow experiment name (required)
        run_name: Specific run identifier (required)
        adapter: Which training script adapter to use (required)
        parameters: Hyperparameters passed to training script (required)
        tags: MLflow tags for organization (required)
        random_seed: Reproducibility seed (required, default 42)
        description: Human-readable experiment description (optional)
        sweep: Parameter sweep definition for CONFIG-03 (optional)

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name="image_biomass_baseline",
        ...     run_name="efficientnet_b0_tta",
        ...     adapter="pytorch",
        ...     parameters={"lr": 0.001, "batch_size": 16},
        ...     tags={"model_type": "cnn"},
        ...     random_seed=42
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'  # Prevent extra fields not in schema
    )

    # Required fields
    experiment_name: str = Field(
        ...,
        min_length=1,
        description="MLflow experiment name"
    )

    run_name: str = Field(
        ...,
        min_length=1,
        description="Specific run identifier"
    )

    adapter: str = Field(
        ...,
        min_length=1,
        description="Which training script adapter to use"
    )

    parameters: Dict[str, Any] = Field(
        ...,
        description="Hyperparameters passed to training script"
    )

    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="MLflow tags for organization"
    )

    random_seed: int = Field(
        default=42,
        gt=0,
        description="Random seed for reproducibility"
    )

    # Optional fields
    description: Optional[str] = Field(
        default=None,
        description="Human-readable experiment description"
    )

    sweep: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameter sweep definition for hyperparameter search"
    )

    @field_validator('experiment_name', 'run_name')
    @classmethod
    def names_must_be_non_empty(cls, v: str) -> str:
        """Validate that experiment_name and run_name are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator('parameters')
    @classmethod
    def parameters_must_be_json_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all parameter values are JSON-serializable.

        This ensures parameters can be logged to MLflow and stored in YAML.
        """
        def is_json_serializable(obj):
            """Check if object is JSON serializable."""
            if obj is None:
                return True
            if isinstance(obj, (str, int, float, bool)):
                return True
            if isinstance(obj, (list, tuple)):
                return all(is_json_serializable(item) for item in obj)
            if isinstance(obj, dict):
                return all(is_json_serializable(v) for v in obj.values())
            return False

        for key, value in v.items():
            if not is_json_serializable(value):
                raise ValueError(
                    f"parameter '{key}' has non-JSON-serializable value: {type(value).__name__}"
                )

        return v

    @field_validator('sweep')
    @classmethod
    def sweep_must_be_valid(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate sweep configuration if provided."""
        if v is None:
            return v

        # Check for required sweep structure
        if 'grid' in v:
            if not isinstance(v['grid'], dict):
                raise ValueError("sweep.grid must be a dictionary")
            if not v['grid']:
                raise ValueError("sweep.grid must not be empty")

        if 'variables' in v:
            if not isinstance(v['variables'], dict):
                raise ValueError("sweep.variables must be a dictionary")

        return v

    def get_sweep_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from sweep definition.

        Returns:
            List of parameter dictionaries, one for each sweep combination.
            Returns single base parameters if no sweep defined.

        Example:
            >>> config = ExperimentConfig(
            ...     experiment_name="test",
            ...     run_name="test",
            ...     adapter="pytorch",
            ...     parameters={"lr": 0.001},
            ...     tags={},
            ...     sweep={"grid": {"lr": [0.001, 0.01], "bs": [8, 16]}}
            ... )
            >>> combos = config.get_sweep_combinations()
            >>> len(combos)
            4
        """
        if self.sweep is None or 'grid' not in self.sweep:
            return [self.parameters.copy()]

        import itertools

        grid = self.sweep['grid']
        keys = list(grid.keys())
        values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            params = self.parameters.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            combinations.append(params)

        return combinations

    def render_run_name(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render run name with Jinja2 template variables.

        Args:
            params: Optional parameters to substitute in template.
                    Uses self.parameters if not provided.

        Returns:
            Rendered run name with variables substituted.

        Example:
            >>> config = ExperimentConfig(
            ...     experiment_name="test",
            ...     run_name="run_lr{{lr}}",
            ...     adapter="pytorch",
            ...     parameters={"lr": 0.001},
            ...     tags={}
            ... )
            >>> config.render_run_name()
            'run_lr0.001'
        """
        from jinja2 import Template

        template_params = params if params is not None else self.parameters

        # Check if run_name contains Jinja2 template syntax
        if '{{' not in self.run_name and '{%' not in self.run_name:
            return self.run_name

        try:
            template = Template(self.run_name)
            return template.render(**template_params)
        except Exception as e:
            raise ValueError(f"Failed to render run_name template: {e}")


class ConfigParser:
    """Parse and validate experiment configuration YAML files.

    Handles:
    - YAML loading and validation against ExperimentConfig schema
    - Jinja2 templating for variable substitution
    - Sweep expansion for grid search
    - Adapter validation (checking adapter exists and config is valid for it)
    """

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load and validate experiment configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validated ExperimentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If config fails schema validation
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Validate against schema
        try:
            config = ExperimentConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")

        return config

    @staticmethod
    def _apply_template(config_yaml: str, variables: Dict[str, Any]) -> str:
        """Apply Jinja2 templating to YAML string.

        Args:
            config_yaml: Raw YAML string with {{variable}} placeholders
            variables: Dict of variable names to values

        Returns:
            Rendered YAML string with variables substituted

        Raises:
            TemplateError: If template syntax is invalid
        """
        template = Template(config_yaml)
        return template.render(**variables)

    @staticmethod
    def _generate_sweep_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search.

        Args:
            grid: Dict of parameter names to value lists
                  e.g., {'lr': [0.001, 0.01], 'bs': [8, 16]}

        Returns:
            List of dicts, one per combination
            e.g., [{'lr': 0.001, 'bs': 8}, {'lr': 0.001, 'bs': 16}, ...]
        """
        keys = grid.keys()
        values = grid.values()
        combinations = itertools.product(*values)

        return [dict(zip(keys, combo)) for combo in combinations]

    @staticmethod
    def expand_sweeps(config_path: Union[str, Path]) -> List[ExperimentConfig]:
        """Expand parameter sweeps into multiple configurations.

        For configs with a 'sweep.grid' section, generates one config per
        combination of grid parameters. Jinja2 templates in run_name and
        parameters are substituted with sweep values.

        Args:
            config_path: Path to YAML configuration file (may contain sweep)

        Returns:
            List of ExperimentConfig instances (one per sweep combination)

        Example:
            Given sweep.grid = {'lr': [0.001, 0.01], 'bs': [8, 16]}
            Returns 4 configs with all (lr, bs) combinations
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load raw YAML string (before parsing)
        with open(config_path) as f:
            config_yaml = f.read()

        # Parse YAML to check for sweep section
        config_dict = yaml.safe_load(config_yaml)

        # If no sweep, return single config
        if 'sweep' not in config_dict or 'grid' not in config_dict['sweep']:
            return [ConfigParser.load_config(config_path)]

        # Generate sweep combinations
        grid = config_dict['sweep']['grid']
        combinations = ConfigParser._generate_sweep_combinations(grid)

        # Remove sweep section from template (not needed in individual configs)
        # Use jinja2 to render config for each combination
        configs = []
        for combo in combinations:
            # Apply template substitution
            rendered_yaml = ConfigParser._apply_template(config_yaml, combo)

            # Parse rendered YAML
            rendered_dict = yaml.safe_load(rendered_yaml)

            # Remove sweep section from individual configs
            if 'sweep' in rendered_dict:
                del rendered_dict['sweep']

            # Validate and create config
            try:
                config = ExperimentConfig(**rendered_dict)
                configs.append(config)
            except Exception as e:
                raise ValueError(f"Sweep config validation failed for combo {combo}: {e}")

        return configs

    @staticmethod
    def validate(config: ExperimentConfig, adapter_registry: 'AdapterRegistry') -> bool:
        """Validate configuration against adapter requirements.

        This method checks:
        1. Adapter exists in registry
        2. Config has all required parameters for the adapter
        3. Parameter types are correct for the adapter

        Args:
            config: Experiment configuration to validate
            adapter_registry: AdapterRegistry instance to check adapter existence

        Returns:
            True if valid

        Raises:
            ValueError: If adapter not registered or config invalid for adapter
        """
        # Check adapter exists in registry
        if config.adapter not in adapter_registry._adapters:
            available = list(adapter_registry._adapters.keys())
            raise ValueError(
                f"Unknown adapter '{config.adapter}'. "
                f"Available adapters: {available}"
            )

        # Validate against adapter-specific requirements
        adapter = adapter_registry.get(config.adapter)
        adapter.validate_config(config)

        return True
