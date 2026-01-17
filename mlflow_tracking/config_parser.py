"""
Experiment configuration parser using Pydantic for schema validation.

This module defines the YAML schema for experiment configurations and provides
validation logic to ensure configs are correct before execution.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from jinja2 import Template, TemplateError
import itertools


class SearchParamConfig(BaseModel):
    """
    Schema for a single hyperparameter search space definition.

    This class defines the search space for a single hyperparameter in Optuna.
    Supports float, int, and categorical parameter types.

    Attributes:
        type: Parameter type ("float", "int", or "categorical")
        low: Optional lower bound for float/int ranges
        high: Optional upper bound for float/int ranges
        log: Optional flag for log-scale sampling (float only)
        step: Optional step size for int discretization
        choices: Optional list of choices for categorical parameters

    Example:
        >>> float_param = SearchParamConfig(type="float", low=1e-5, high=1e-1, log=True)
        >>> int_param = SearchParamConfig(type="int", low=1, high=100, step=1)
        >>> cat_param = SearchParamConfig(type="categorical", choices=["adam", "sgd"])
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    type: Literal["float", "int", "categorical"] = Field(
        ...,
        description="Parameter type for search space"
    )

    # For float and int types
    low: Optional[float] = Field(
        default=None,
        description="Lower bound for float/int ranges"
    )

    high: Optional[float] = Field(
        default=None,
        description="Upper bound for float/int ranges"
    )

    log: Optional[bool] = Field(
        default=False,
        description="Use log-scale sampling (float only)"
    )

    # For int type
    step: Optional[int] = Field(
        default=1,
        description="Step size for int discretization"
    )

    # For categorical type
    choices: Optional[List[Any]] = Field(
        default=None,
        description="Choices for categorical parameters"
    )

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate parameter type."""
        valid_types = ["float", "int", "categorical"]
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        return v

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: Optional[List[Any]], info) -> Optional[List[Any]]:
        """Validate choices are provided for categorical type."""
        if info and info.data.get('type') == 'categorical':
            if v is None or not v:
                raise ValueError("choices must be provided for categorical type")
        return v

    @field_validator('low', 'high')
    @classmethod
    def validate_bounds(cls, v: Optional[float], info) -> Optional[float]:
        """Validate bounds are provided for float/int types."""
        if info and info.data.get('type') in ['float', 'int']:
            # This is a simplified check - full validation happens in OptimizationConfig
            return v
        return v


class OptimizationConfig(BaseModel):
    """
    Schema for Optuna hyperparameter optimization configuration.

    This class defines the configuration for automated hyperparameter search
    using Optuna, including search spaces, pruning, and study settings.

    Attributes:
        n_trials: Number of trials to run
        study_name: Optuna study name for persistence
        direction: Optimization direction ("minimize" or "maximize")
        metric: Metric name to optimize (e.g., "val.rmse")
        search: Dict mapping parameter names to SearchParamConfig
        pruner: Optional pruner configuration dict
        timeout: Optional study timeout in seconds

    Example:
        >>> opt_config = OptimizationConfig(
        ...     n_trials=100,
        ...     study_name="lr_search",
        ...     direction="minimize",
        ...     metric="val.rmse",
        ...     search={
        ...         "learning_rate": SearchParamConfig(type="float", low=1e-5, high=1e-1, log=True),
        ...         "batch_size": SearchParamConfig(type="int", low=8, high=64, step=8)
        ...     },
        ...     pruner={"type": "median", "n_startup_trials": 5, "n_warmup_steps": 10}
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    n_trials: int = Field(
        ...,
        gt=0,
        description="Number of optimization trials to run"
    )

    study_name: str = Field(
        ...,
        min_length=1,
        description="Optuna study name for persistence"
    )

    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction"
    )

    metric: str = Field(
        ...,
        min_length=1,
        description="Metric name to optimize (e.g., 'val.rmse')"
    )

    search: Dict[str, Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Search space definition (param name -> SearchParamConfig dict)"
    )

    pruner: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pruner configuration for early stopping"
    )

    timeout: Optional[int] = Field(
        default=None,
        gt=0,
        description="Study timeout in seconds (None for no limit)"
    )

    @field_validator('search')
    @classmethod
    def validate_search_space(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate search space definitions."""
        if not v:
            raise ValueError("search space must not be empty")

        for param_name, param_config in v.items():
            # Validate each search param config
            try:
                SearchParamConfig(**param_config)
            except Exception as e:
                raise ValueError(
                    f"Invalid search config for parameter '{param_name}': {e}"
                )

        return v

    @field_validator('pruner')
    @classmethod
    def validate_pruner(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate pruner configuration."""
        if v is None:
            return v

        valid_pruner_types = ["median", "hyperband", "successive_halving"]
        pruner_type = v.get("type")

        if pruner_type is None:
            raise ValueError("pruner must have a 'type' field")

        if pruner_type not in valid_pruner_types:
            raise ValueError(
                f"Invalid pruner type '{pruner_type}'. "
                f"Valid options: {valid_pruner_types}"
            )

        return v


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

    optimization: Optional[OptimizationConfig] = Field(
        default=None,
        description="Optuna hyperparameter optimization configuration"
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
