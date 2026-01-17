"""
Optuna hyperparameter optimization with MLflow integration.

This module provides the OptunaOptimizer class for automated hyperparameter
search using Optuna with MLflow logging for experiment tracking.
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from optuna.integration import MLflowCallback
from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path
import yaml

from mlflow_tracking.config_parser import ExperimentConfig, OptimizationConfig, SearchParamConfig
from mlflow_tracking.tracker import ExperimentTracker
from mlflow_tracking.adapters import AdapterRegistry


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer with MLflow integration.

    This class manages Optuna studies for automated hyperparameter search,
    integrating with the existing experiment tracking infrastructure via
    ExperimentTracker and training script adapters.

    Attributes:
        optimization_config: Optimization configuration (search space, trials, etc.)
        base_config: Base experiment configuration (adapter, tags, etc.)
        tracker: MLflow experiment tracker
        study: Optuna study object (created after run_study())

    Example:
        >>> optimizer = OptunaOptimizer(optimization_config, base_config, tracker)
        >>> study = optimizer.run_study(n_jobs=1)
        >>> best_params = optimizer.get_best_params()
        >>> best_config = optimizer.generate_best_config()
    """

    def __init__(
        self,
        optimization_config: OptimizationConfig,
        base_config: ExperimentConfig,
        tracker: ExperimentTracker,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            optimization_config: Optimization configuration with search space
            base_config: Base experiment configuration to use for trials
            tracker: MLflow experiment tracker for logging

        Raises:
            ValueError: If optimization_config is invalid
        """
        if not isinstance(optimization_config, OptimizationConfig):
            raise ValueError(
                f"optimization_config must be OptimizationConfig, "
                f"got {type(optimization_config)}"
            )

        self.optimization_config = optimization_config
        self.base_config = base_config
        self.tracker = tracker
        self.study: Optional[optuna.Study] = None

        # Parse search space for validation
        self._search_space = self._parse_search_space()

    def _parse_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse and validate search space from optimization config.

        Returns:
            Dict mapping parameter names to their SearchParamConfig dicts

        Raises:
            ValueError: If search space is invalid or empty
        """
        search = self.optimization_config.search

        if not search:
            raise ValueError("Search space must not be empty")

        # Validate each parameter config
        for param_name, param_config in search.items():
            try:
                SearchParamConfig(**param_config)
            except Exception as e:
                raise ValueError(
                    f"Invalid search config for parameter '{param_name}': {e}"
                )

        return search

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """
        Create Optuna pruner from configuration.

        Returns:
            Optuna pruner instance

        Raises:
            ValueError: If pruner type is invalid
        """
        pruner_config = self.optimization_config.pruner

        if pruner_config is None:
            # Default: MedianPruner with sensible defaults
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        pruner_type = pruner_config.get("type")

        if pruner_type == "median":
            return MedianPruner(
                n_startup_trials=pruner_config.get("n_startup_trials", 5),
                n_warmup_steps=pruner_config.get("n_warmup_steps", 10),
                interval_steps=pruner_config.get("interval_steps", 1),
            )
        elif pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=pruner_config.get("min_resource", 1),
                max_resource=pruner_config.get("max_resource", 100),
                reduction_factor=pruner_config.get("reduction_factor", 3),
            )
        elif pruner_type == "successive_halving":
            return SuccessiveHalvingPruner(
                reduction_factor=pruner_config.get("reduction_factor", 4),
                min_early_stopping_rate=pruner_config.get("min_early_stopping_rate", 0),
            )
        else:
            raise ValueError(
                f"Unknown pruner type '{pruner_type}'. "
                f"Valid options: median, hyperband, successive_halving"
            )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        This method is called by Optuna for each trial. It:
        1. Suggests hyperparameters from the search space
        2. Creates a trial-specific config
        3. Executes the training script via adapter
        4. Returns the metric value to optimize

        Args:
            trial: Optuna trial object for parameter suggestion

        Returns:
            Metric value to optimize (float)

        Raises:
            Exception: If trial execution fails (returns inf for minimization)
        """
        # Suggest parameters from search space
        trial_params = suggest_params_from_trial(trial, self._search_space)

        # Merge with base parameters
        trial_config_params = self.base_config.parameters.copy()
        trial_config_params.update(trial_params)

        # Create trial-specific config
        trial_config = ExperimentConfig(
            experiment_name=self.base_config.experiment_name,
            run_name=f"{self.base_config.run_name}_trial_{trial.number}",
            adapter=self.base_config.adapter,
            parameters=trial_config_params,
            tags={**self.base_config.tags, "trial_number": str(trial.number)},
            random_seed=self.base_config.random_seed,
            description=self.base_config.description,
        )

        # Get adapter and execute
        adapter = AdapterRegistry.get(trial_config.adapter)

        try:
            # Start MLflow run for this trial
            run_id = self.tracker.start_run(
                run_name=trial_config.run_name,
                tags=trial_config.tags,
                random_seed=trial_config.random_seed,
            )

            # Log parameters
            self.tracker.log_params(trial_config.parameters)

            # Execute training
            metrics = adapter.execute(trial_config, self.tracker)

            # Extract the metric to optimize
            metric_name = self.optimization_config.metric
            if metric_name not in metrics:
                raise ValueError(
                    f"Metric '{metric_name}' not found in trial results. "
                    f"Available metrics: {list(metrics.keys())}"
                )

            metric_value = metrics[metric_name]

            # Log final metrics
            self.tracker.log_metrics(metrics)

            # End run successfully
            self.tracker.end_run(status="completed")

            return float(metric_value)

        except Exception as e:
            # Mark run as failed
            if self.tracker.active_run:
                self.tracker.mark_failed(error_message=str(e))

            # Return inf for minimization (Optuna will handle this)
            print(f"Trial {trial.number} failed: {e}")
            return float("inf") if self.optimization_config.direction == "minimize" else float("-inf")

    def run_study(
        self,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> optuna.Study:
        """
        Run the Optuna optimization study.

        Args:
            n_jobs: Number of parallel trials (-1 for auto-detect)
            show_progress: Whether to show progress bar

        Returns:
            Optuna study object with completed trials

        Raises:
            Exception: If study creation or optimization fails
        """
        # Create MLflow callback for logging trials to MLflow
        mlflow_callback = MLflowCallback(
            tracking_uri=self.tracker.client.tracking_uri,
            metric_name=self.optimization_config.metric,
        )

        # Create study
        self.study = optuna.create_study(
            study_name=self.optimization_config.study_name,
            direction=self.optimization_config.direction,
            pruner=self._create_pruner(),
            load_if_exists=True,  # Resume existing study if present
        )

        # Run optimization
        print(f"Starting optimization study: {self.optimization_config.study_name}")
        print(f"Trials: {self.optimization_config.n_trials}")
        print(f"Direction: {self.optimization_config.direction}")
        print(f"Metric: {self.optimization_config.metric}")
        print(f"Search space: {list(self._search_space.keys())}")

        self.study.optimize(
            self.objective,
            n_trials=self.optimization_config.n_trials,
            n_jobs=n_jobs,
            callbacks=[mlflow_callback],
            show_progress_bar=show_progress,
            timeout=self.optimization_config.timeout,
        )

        print(f"\nStudy complete!")
        print(f"Best value: {self.study.best_value}")
        print(f"Best params: {self.study.best_params}")

        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters from optimization study.

        Returns:
            Dict of best hyperparameter values

        Raises:
            RuntimeError: If study hasn't been run yet
        """
        if self.study is None:
            raise RuntimeError("Study hasn't been run yet. Call run_study() first.")

        return self.study.best_params

    def get_best_trial(self) -> optuna.Trial:
        """
        Get best trial from optimization study.

        Returns:
            Best Optuna trial object

        Raises:
            RuntimeError: If study hasn't been run yet
        """
        if self.study is None:
            raise RuntimeError("Study hasn't been run yet. Call run_study() first.")

        return self.study.best_trial

    def generate_best_config(self) -> ExperimentConfig:
        """
        Generate experiment config with best hyperparameters.

        Creates a new ExperimentConfig with the best parameters from
        optimization, suitable for running a final experiment with
        optimal hyperparameters.

        Returns:
            ExperimentConfig with best hyperparameters

        Raises:
            RuntimeError: If study hasn't been run yet
        """
        if self.study is None:
            raise RuntimeError("Study hasn't been run yet. Call run_study() first.")

        best_params = self.get_best_params()

        # Merge best params with base config
        best_config_params = self.base_config.parameters.copy()
        best_config_params.update(best_params)

        # Create best config
        best_config = ExperimentConfig(
            experiment_name=self.base_config.experiment_name,
            run_name=f"{self.base_config.run_name}_best",
            adapter=self.base_config.adapter,
            parameters=best_config_params,
            tags={**self.base_config.tags, "optimized": "true"},
            random_seed=self.base_config.random_seed,
            description=f"Best config from {self.optimization_config.study_name}",
        )

        return best_config


def create_optimization_config(config_dict: Dict[str, Any]) -> OptimizationConfig:
    """
    Create OptimizationConfig from dictionary.

    Helper function to create OptimizationConfig from a dict (e.g.,
    loaded from YAML). This is useful for programmatic config creation.

    Args:
        config_dict: Dictionary with optimization config fields

    Returns:
        Validated OptimizationConfig instance

    Raises:
        ValueError: If config is invalid

    Example:
        >>> opt_dict = {
        ...     "n_trials": 100,
        ...     "study_name": "lr_search",
        ...     "direction": "minimize",
        ...     "metric": "val.rmse",
        ...     "search": {
        ...         "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": true}
        ...     }
        ... }
        >>> opt_config = create_optimization_config(opt_dict)
    """
    try:
        return OptimizationConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid optimization config: {e}")


def suggest_params_from_trial(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Suggest hyperparameters from Optuna trial based on search space.

    Helper function to suggest parameters from a trial using the
    search space definition. This is used by OptunaOptimizer.objective
    but can also be used standalone.

    Args:
        trial: Optuna trial object
        search_space: Dict mapping parameter names to SearchParamConfig dicts

    Returns:
        Dict of suggested parameter values

    Raises:
        ValueError: If parameter type or config is invalid

    Example:
        >>> search_space = {
        ...     "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        ...     "bs": {"type": "int", "low": 8, "high": 64, "step": 8}
        ... }
        >>> params = suggest_params_from_trial(trial, search_space)
        >>> # params = {"lr": 0.001, "bs": 32}
    """
    params = {}

    for param_name, param_config in search_space.items():
        param_type = param_config.get("type")

        if param_type == "float":
            low = param_config.get("low")
            high = param_config.get("high")
            log = param_config.get("log", False)

            if low is None or high is None:
                raise ValueError(
                    f"Float parameter '{param_name}' requires 'low' and 'high' bounds"
                )

            params[param_name] = trial.suggest_float(param_name, low, high, log=log)

        elif param_type == "int":
            low = param_config.get("low")
            high = param_config.get("high")
            step = param_config.get("step", 1)

            if low is None or high is None:
                raise ValueError(
                    f"Int parameter '{param_name}' requires 'low' and 'high' bounds"
                )

            params[param_name] = trial.suggest_int(param_name, low, high, step=step)

        elif param_type == "categorical":
            choices = param_config.get("choices")

            if choices is None:
                raise ValueError(
                    f"Categorical parameter '{param_name}' requires 'choices'"
                )

            params[param_name] = trial.suggest_categorical(param_name, choices)

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for '{param_name}'. "
                f"Valid types: float, int, categorical"
            )

    return params
