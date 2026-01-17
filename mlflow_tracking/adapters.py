"""
Abstract adapter interface for wrapping training scripts.

This module provides the BaseAdapter protocol and AdapterRegistry for
registering and retrieving training script adapters, enabling existing
scripts to run via YAML config without modification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type
import subprocess
import json

from mlflow_tracking.config_parser import ExperimentConfig


class BaseAdapter(ABC):
    """
    Abstract base class for training script adapters.

    Adapters enable existing training scripts to run via YAML config
    without modifying the original scripts (INTEGRATION-01).

    Each adapter is responsible for:
    1. Validating that the config has required parameters for the script
    2. Executing the training script with the given configuration
    3. Returning metrics to log to MLflow

    Example:
        >>> @AdapterRegistry.register('pytorch')
        >>> class PyTorchAdapter(BaseAdapter):
        >>>     def validate_config(self, config):
        >>>         required = ['model_name', 'batch_size', 'epochs']
        >>>         for param in required:
        >>>             if param not in config.parameters:
        >>>                 raise ValueError(f"Missing required parameter: {param}")
        >>>         return True
        >>>
        >>>     def execute(self, config, tracker):
        >>>         # Execute training script
        >>>         metrics = train_model(config.parameters)
        >>>         return {'train.rmse': metrics['train_rmse']}
    """

    @abstractmethod
    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        """
        Execute training script with given configuration.

        This method should:
        1. Start a MLflow run via tracker (if not already started)
        2. Log parameters from config.parameters
        3. Execute the training script with those parameters
        4. Log metrics during/after training
        5. Return final metrics as a dictionary

        Args:
            config: Validated experiment configuration
            tracker: MLflow experiment tracker for logging

        Returns:
            Dictionary of metrics to log (e.g., {'train.rmse': 8.2, 'val.rmse': 10.5})

        Raises:
            Exception: Script execution failures (tracker will mark as FAILED)

        Note:
            The tracker may have an active run or not. Adapters should
            check tracker.active_run and call tracker.start_run() if needed.
        """
        pass

    @abstractmethod
    def validate_config(self, config: ExperimentConfig) -> bool:
        """
        Validate that config has required parameters for this adapter.

        Each adapter type has different required parameters. This method
        should verify that all required parameters are present and valid.

        Args:
            config: Experiment configuration to validate

        Returns:
            True if config is valid for this adapter

        Raises:
            ValueError: If config missing required parameters or has invalid values

        Example:
            >>> def validate_config(self, config):
            >>>     required = ['model_name', 'batch_size', 'epochs', 'learning_rate']
            >>>     missing = [p for p in required if p not in config.parameters]
            >>>     if missing:
            >>>         raise ValueError(f"Missing required parameters: {missing}")
            >>>     if config.parameters['batch_size'] <= 0:
            >>>         raise ValueError("batch_size must be positive")
            >>>     return True
        """
        pass


class AdapterRegistry:
    """
    Registry for training script adapters.

    Provides a decorator-based registration system and retrieval
    of adapter instances by name.

    Attributes:
        _adapters: Class-level dictionary mapping names to adapter classes

    Example:
        >>> @AdapterRegistry.register('pytorch')
        >>> class PyTorchAdapter(BaseAdapter):
        >>>     def execute(self, config, tracker):
        >>>         ...
        >>>     def validate_config(self, config):
        >>>         ...
        >>>
        >>> adapter = AdapterRegistry.get('pytorch')
        >>> adapter.execute(config, tracker)
    """

    _adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator for registering adapter classes.

        Usage:
            @AdapterRegistry.register('pytorch')
            class PyTorchAdapter(BaseAdapter):
                ...

        Args:
            name: Unique name for this adapter (referenced in YAML configs)

        Returns:
            Decorator function that registers the class

        Raises:
            ValueError: If adapter name already registered
        """
        def decorator(adapter_class: Type[BaseAdapter]):
            if name in cls._adapters:
                raise ValueError(
                    f"Adapter '{name}' already registered. "
                    f"Use a different name or unregister first."
                )
            if not issubclass(adapter_class, BaseAdapter):
                raise TypeError(
                    f"Adapter '{name}' must inherit from BaseAdapter"
                )
            cls._adapters[name] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def get(cls, name: str) -> BaseAdapter:
        """
        Get adapter instance by name.

        Args:
            name: Registered adapter name

        Returns:
            Instance of the requested adapter

        Raises:
            ValueError: If adapter name not registered

        Example:
            >>> adapter = AdapterRegistry.get('pytorch')
            >>> isinstance(adapter, PyTorchAdapter)
            True
        """
        if name not in cls._adapters:
            available = list(cls._adapters.keys())
            raise ValueError(
                f"Unknown adapter: '{name}'. "
                f"Available adapters: {available if available else '(none)'}"
            )
        return cls._adapters[name]()

    @classmethod
    def list_adapters(cls) -> list[str]:
        """
        List all registered adapter names.

        Returns:
            List of registered adapter names

        Example:
            >>> AdapterRegistry.list_adapters()
            ['pytorch', 'sklearn', 'custom']
        """
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an adapter is registered.

        Args:
            name: Adapter name to check

        Returns:
            True if adapter is registered, False otherwise
        """
        return name in cls._adapters

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an adapter by name.

        Useful for testing or replacing adapters.

        Args:
            name: Adapter name to unregister

        Raises:
            ValueError: If adapter name not registered
        """
        if name not in cls._adapters:
            raise ValueError(
                f"Cannot unregister unknown adapter: '{name}'"
            )
        del cls._adapters[name]


@AdapterRegistry.register('pytorch')
class PyTorchAdapter(BaseAdapter):
    """Adapter for PyTorch-based training scripts.

    Invokes training scripts via subprocess, passing parameters as command-line args.
    Scripts must output metrics as JSON to stdout for logging.

    Example script invocation:
        python scripts/train_oof_effnet.py \\
            --model_name efficientnet_b0 \\
            --batch_size 16 \\
            --epochs 30 \\
            --learning_rate 0.0001

    Expected JSON output:
        {"train_rmse": 8.2, "val_rmse": 10.5, "test_rmse": 11.3}

    Pattern for wrapping additional PyTorch scripts:
    1. Create new adapter class: @AdapterRegistry.register('pytorch_resnet')
    2. Update validate_config() for required parameters
    3. Update script_path in execute() method
    4. Update param_mapping if CLI args differ
    That's it - same pattern for all PyTorch scripts.
    """

    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate PyTorch-specific parameters.

        Required parameters:
            model_name: str (e.g., 'efficientnet_b0', 'resnet18')
            batch_size: int
            epochs: int
            learning_rate: float

        Optional parameters:
            image_size: int (default 224)
            use_tta: bool (default false)
            pretrained: bool (default true)
        """
        required = ['model_name', 'batch_size', 'epochs', 'learning_rate']

        for param in required:
            if param not in config.parameters:
                raise ValueError(
                    f"PyTorchAdapter requires parameter '{param}'. "
                    f"Config has: {list(config.parameters.keys())}"
                )

        # Type validation
        if not isinstance(config.parameters['batch_size'], int):
            raise ValueError("batch_size must be an integer")
        if not isinstance(config.parameters['epochs'], int):
            raise ValueError("epochs must be an integer")
        if not isinstance(config.parameters['learning_rate'], (int, float)):
            raise ValueError("learning_rate must be a number")

        return True

    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        """Execute PyTorch training script.

        Args:
            config: Experiment configuration with script parameters
            tracker: MLflow tracker for logging (available for intermediate logging if needed)

        Returns:
            Dictionary of final metrics (e.g., {'train.rmse': 8.2, 'val.rmse': 10.5})
            Note: Metrics are logged to MLflow by CLI, not by adapter directly.

        Raises:
            subprocess.CalledProcessError: If script fails
            ValueError: If script doesn't output JSON metrics
        """
        # Map config parameters to command-line arguments
        script_path = "scripts/train_oof_effnet.py"

        # Build command args from config parameters
        args = ["python3", script_path]

        # Convert parameters to CLI args
        param_mapping = {
            'model_name': '--model_name',
            'batch_size': '--batch_size',
            'epochs': '--epochs',
            'learning_rate': '--learning_rate',
            'image_size': '--image_size',
            'use_tta': '--use_tta',
            'pretrained': '--pretrained',
        }

        for param_name, arg_name in param_mapping.items():
            if param_name in config.parameters:
                value = config.parameters[param_name]
                args.append(arg_name)
                args.append(str(value))

        # Execute script
        print(f"Executing: {' '.join(args)}")
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse JSON output from script
        # Assume script prints metrics as JSON: {"train_rmse": 8.2, "val_rmse": 10.5}
        try:
            # Last non-empty line should be JSON
            output_lines = [line for line in result.stdout.split('\n') if line.strip()]
            json_output = output_lines[-1] if output_lines else "{}"
            metrics_dict = json.loads(json_output)

            # Convert to MLflow format (use dots for hierarchy)
            metrics = {}
            for key, value in metrics_dict.items():
                # Convert train_rmse -> train.rmse
                mlflow_key = key.replace('_', '.')
                metrics[mlflow_key] = float(value)

            return metrics

        except (json.JSONDecodeError, IndexError, ValueError) as e:
            raise ValueError(
                f"Script output parsing failed. "
                f"Expected JSON metrics on last line of stdout.\n"
                f"Error: {e}\n"
                f"Output: {result.stdout}"
            )


@AdapterRegistry.register('sklearn')
class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn based training scripts.

    Invokes training scripts via subprocess, passing parameters as command-line args.
    Scripts must output metrics as JSON to stdout for logging.

    Example script invocation:
        python scripts/train_ridge_advanced.py \\
            --alpha 1.0 \\
            --fit_intercept true \\
            --random_seed 42

    Expected JSON output:
        {"train_rmse": 8.2, "val_rmse": 10.5, "train_r2": 0.85, "val_r2": 0.78}

    Pattern for wrapping additional sklearn scripts:
    1. Create new adapter class: @AdapterRegistry.register('sklearn_xgboost')
    2. Update validate_config() for required parameters
    3. Update script_path in execute() method
    That's it - same pattern for all sklearn scripts.
    """

    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate sklearn-specific parameters.

        Required parameters:
            model_type: str (e.g., 'ridge', 'lasso', 'xgboost')
            random_seed: int

        Optional parameters (model-dependent):
            alpha: float (for Ridge, Lasso)
            n_estimators: int (for XGBoost, Random Forest)
            max_depth: int (for tree-based models)
            fit_intercept: bool (for linear models)
        """
        required = ['model_type', 'random_seed']

        for param in required:
            if param not in config.parameters:
                raise ValueError(
                    f"SklearnAdapter requires parameter '{param}'. "
                    f"Config has: {list(config.parameters.keys())}"
                )

        # Validate model type
        valid_models = ['ridge', 'lasso', 'xgboost', 'random_forest']
        if config.parameters['model_type'] not in valid_models:
            raise ValueError(
                f"Invalid model_type '{config.parameters['model_type']}'. "
                f"Valid options: {valid_models}"
            )

        return True

    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        """Execute sklearn training script.

        Args:
            config: Experiment configuration with script parameters
            tracker: MLflow tracker for logging (available for intermediate logging if needed)

        Returns:
            Dictionary of final metrics (e.g., {'train.rmse': 8.2, 'val.rmse': 10.5})
            Note: Metrics are logged to MLflow by CLI, not by adapter directly.

        Raises:
            subprocess.CalledProcessError: If script fails
            ValueError: If script doesn't output JSON metrics
        """
        # Map config parameters to command-line arguments
        script_path = "scripts/train_ridge_advanced.py"

        # Build command args from config parameters
        args = ["python3", script_path]

        # Convert all parameters to CLI args
        for param_name, value in config.parameters.items():
            arg_name = f"--{param_name}"
            args.append(arg_name)
            args.append(str(value))

        # Execute script
        print(f"Executing: {' '.join(args)}")
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse JSON output from script
        try:
            # Last non-empty line should be JSON
            output_lines = [line for line in result.stdout.split('\n') if line.strip()]
            json_output = output_lines[-1] if output_lines else "{}"
            metrics_dict = json.loads(json_output)

            # Convert to MLflow format (use dots for hierarchy)
            metrics = {}
            for key, value in metrics_dict.items():
                # Convert train_rmse -> train.rmse
                mlflow_key = key.replace('_', '.')
                metrics[mlflow_key] = float(value)

            return metrics

        except (json.JSONDecodeError, IndexError, ValueError) as e:
            raise ValueError(
                f"Script output parsing failed. "
                f"Expected JSON metrics on last line of stdout.\n"
                f"Error: {e}\n"
                f"Output: {result.stdout}"
            )
