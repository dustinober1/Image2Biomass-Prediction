"""
Abstract adapter interface for wrapping training scripts.

This module provides the BaseAdapter protocol and AdapterRegistry for
registering and retrieving training script adapters, enabling existing
scripts to run via YAML config without modification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

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
