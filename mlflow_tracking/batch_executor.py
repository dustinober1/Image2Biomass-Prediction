"""
Batch executor for parallel experiment execution.

This module provides BatchExecutor class for running multiple experiments
in parallel with automatic resource management and progress monitoring.
"""

import os
import time
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from mlflow_tracking.config_parser import ExperimentConfig, ConfigParser
from mlflow_tracking.resource_manager import ResourceManager, ResourceToken
from mlflow_tracking.tracker import ExperimentTracker
from mlflow_tracking.adapters import AdapterRegistry


@dataclass
class ExperimentResult:
    """
    Result of a single experiment execution.

    Attributes:
        run_id: MLflow run ID (None if execution failed before run start)
        status: Execution status ('completed', 'failed', 'pending')
        metrics: Dictionary of metrics (empty if failed)
        error: Error message (None if successful)
        config: Experiment configuration used
        duration: Execution duration in seconds
    """
    run_id: Optional[str]
    status: str
    metrics: Dict[str, float]
    error: Optional[str]
    config: ExperimentConfig
    duration: float = 0.0


@dataclass
class BatchProgress:
    """
    Progress tracking for batch execution.

    Attributes:
        total: Total number of experiments
        completed: Number of completed experiments
        failed: Number of failed experiments
        running: Number of currently running experiments
        pending: Number of pending experiments
    """
    total: int
    completed: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0

    def __str__(self) -> str:
        """Return progress summary string."""
        return (
            f"Running {self.running}/{self.total} experiments "
            f"({self.completed} completed, {self.failed} failed, {self.pending} pending)"
        )


class BatchExecutor:
    """
    Execute multiple experiments in parallel with resource management.

    This class provides batch execution capabilities for running multiple
    experiment configurations concurrently while managing GPU/CPU resources
    and monitoring progress.

    Attributes:
        resource_manager: ResourceManager instance for resource allocation
        adapter_registry: AdapterRegistry for retrieving adapters

    Example:
        >>> executor = BatchExecutor()
        >>> configs = executor.load_configs_from_dir("examples/configs/batch/")
        >>> results = executor.execute_batch(configs, max_workers=4, verbose=True)
        >>> for result in results:
        ...     print(f"{result.config.run_name}: {result.status}")
    """

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        adapter_registry: Optional[AdapterRegistry] = None
    ):
        """
        Initialize BatchExecutor.

        Args:
            resource_manager: ResourceManager instance (defaults to singleton)
            adapter_registry: AdapterRegistry instance (defaults to global)
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.adapter_registry = adapter_registry or AdapterRegistry()

    def load_configs(self, config_paths: List[str]) -> List[ExperimentConfig]:
        """
        Load multiple experiment configurations from file paths.

        Args:
            config_paths: List of paths to YAML configuration files

        Returns:
            List of ExperimentConfig instances

        Raises:
            FileNotFoundError: If any config file not found
            ValueError: If any config fails validation
        """
        configs = []
        for path in config_paths:
            try:
                config = ConfigParser.load_config(path)
                configs.append(config)
            except FileNotFoundError as e:
                print(f"Warning: Skipping {path} - {e}")
            except ValueError as e:
                print(f"Warning: Skipping {path} - {e}")

        return configs

    def load_configs_from_dir(
        self,
        dir_path: str,
        pattern: str = "*.yaml"
    ) -> List[ExperimentConfig]:
        """
        Load all YAML configs from a directory.

        Args:
            dir_path: Path to directory containing config files
            pattern: Glob pattern for matching files (default: "*.yaml")

        Returns:
            List of ExperimentConfig instances
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        config_files = sorted(dir_path.glob(pattern))
        return self.load_configs([str(f) for f in config_files])

    def _execute_single_experiment(
        self,
        config: ExperimentConfig,
        gpu_id: Optional[int] = None,
        tracker: Optional[ExperimentTracker] = None
    ) -> ExperimentResult:
        """
        Execute a single experiment with error handling.

        Args:
            config: Experiment configuration
            gpu_id: GPU ID to use (if allocated)
            tracker: ExperimentTracker instance (creates new if None)

        Returns:
            ExperimentResult with execution outcome
        """
        start_time = time.time()
        run_id = None
        metrics = {}
        error = None
        status = 'pending'

        # Add gpu_id to config parameters if provided
        if gpu_id is not None:
            config.parameters['gpu_id'] = gpu_id

        try:
            # Validate config against adapter
            ConfigParser.validate(config, self.adapter_registry)

            # Create tracker if not provided
            if tracker is None:
                tracker = ExperimentTracker(
                    config.experiment_name,
                    auto_log_environment=True
                )

            # Start run
            run_id = tracker.start_run(
                config.run_name,
                tags=config.tags,
                random_seed=config.random_seed
            )

            # Log parameters
            tracker.log_params(config.parameters)

            # Get adapter and execute
            adapter = self.adapter_registry.get(config.adapter)
            metrics = adapter.execute(config, tracker)

            # Log metrics
            if metrics:
                tracker.log_metrics(metrics)

            status = 'completed'

        except Exception as e:
            error = str(e)
            status = 'failed'
            if tracker and tracker.active_run:
                try:
                    tracker.mark_failed(error)
                except Exception:
                    pass  # Already failed, ignore

        duration = time.time() - start_time

        return ExperimentResult(
            run_id=run_id,
            status=status,
            metrics=metrics,
            error=error,
            config=config,
            duration=duration
        )

    def _execute_with_resource_management(
        self,
        config: ExperimentConfig,
        resource_token: Optional[ResourceToken] = None,
        tracker: Optional[ExperimentTracker] = None
    ) -> ExperimentResult:
        """
        Execute experiment with resource token (if provided).

        Args:
            config: Experiment configuration
            resource_token: ResourceToken from ResourceManager
            tracker: ExperimentTracker instance

        Returns:
            ExperimentResult with execution outcome
        """
        gpu_id = resource_token.gpu_id if resource_token else None
        return self._execute_single_experiment(config, gpu_id, tracker)

    def execute_batch(
        self,
        configs: List[ExperimentConfig],
        max_workers: Optional[int] = None,
        verbose: bool = False,
        on_start: Optional[Callable[[ExperimentConfig], None]] = None,
        on_complete: Optional[Callable[[ExperimentResult], None]] = None,
        on_error: Optional[Callable[[ExperimentResult], None]] = None
    ) -> List[ExperimentResult]:
        """
        Execute multiple experiments in parallel.

        Args:
            configs: List of experiment configurations
            max_workers: Maximum number of concurrent experiments (auto-suggest if None)
            verbose: Print progress information
            on_start: Callback when experiment starts
            on_complete: Callback when experiment completes
            on_error: Callback when experiment fails

        Returns:
            List of ExperimentResult in same order as input configs
        """
        if not configs:
            if verbose:
                print("No configs to execute")
            return []

        # Auto-suggest max_workers if not provided
        if max_workers is None:
            max_workers = self.resource_manager.suggest_concurrent_experiments()

        # Clamp max_workers to config count
        max_workers = min(max_workers, len(configs))

        # Initialize progress tracking
        progress = BatchProgress(total=len(configs))
        progress.pending = len(configs)

        if verbose:
            summary = self.resource_manager.get_resource_summary()
            print(f"\nAvailable Resources:")
            print(f"  GPUs: {summary['available_gpus']}/{summary['total_gpus']}")
            print(f"  CPUs: {summary['available_cpus']}/{summary['total_cpus']} ({summary['reserved_cpus']} reserved)")
            print(f"Max concurrent experiments: {max_workers}")
            print()

        # Thread-safe progress updates
        progress_lock = threading.Lock()

        def update_progress(
            completed: int = 0,
            failed: int = 0,
            running: int = 0,
            pending: int = 0
        ):
            with progress_lock:
                progress.completed += completed
                progress.failed += failed
                progress.running += running
                progress.pending += pending

        def execute_with_callbacks(config: ExperimentConfig) -> ExperimentResult:
            """Execute experiment with callbacks and progress tracking."""
            update_progress(running=1, pending=-1)

            if verbose:
                print(f"Starting: {config.run_name}")

            if on_start:
                try:
                    on_start(config)
                except Exception:
                    pass

            # Execute experiment
            result = self._execute_with_resource_management(config)

            update_progress(running=-1)
            if result.status == 'completed':
                update_progress(completed=1)
                if verbose:
                    metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in result.metrics.items() if isinstance(v, float))
                    print(f"Completed: {config.run_name} ({metrics_str})")
                if on_complete:
                    try:
                        on_complete(result)
                    except Exception:
                        pass
            else:
                update_progress(failed=1)
                if verbose:
                    print(f"Failed: {config.run_name} - {result.error}")
                if on_error:
                    try:
                        on_error(result)
                    except Exception:
                        pass

            if verbose:
                print(f"  {progress}")

            return result

        # Execute experiments in parallel
        results = []
        result_dict: Dict[str, ExperimentResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(execute_with_callbacks, config): config
                for config in configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    result_dict[config.run_name] = result
                except Exception as e:
                    # Create failed result for unhandled exceptions
                    result = ExperimentResult(
                        run_id=None,
                        status='failed',
                        metrics={},
                        error=str(e),
                        config=config,
                        duration=0.0
                    )
                    result_dict[config.run_name] = result
                    update_progress(failed=1)

        # Return results in same order as input configs
        for config in configs:
            results.append(result_dict.get(config.run_name, ExperimentResult(
                run_id=None,
                status='failed',
                metrics={},
                error='Result not found',
                config=config,
                duration=0.0
            )))

        if verbose:
            print(f"\nBatch complete: {progress.completed}/{progress.total} succeeded, {progress.failed} failed")

        return results

    def get_best_result(
        self,
        results: List[ExperimentResult],
        metric: str = 'val.rmse',
        minimize: bool = True
    ) -> Optional[ExperimentResult]:
        """
        Get the best result from a batch based on a metric.

        Args:
            results: List of experiment results
            metric: Metric name to compare (e.g., 'val.rmse')
            minimize: If True, lower is better (e.g., RMSE)

        Returns:
            Best ExperimentResult or None if no successful results
        """
        successful = [r for r in results if r.status == 'completed' and metric in r.metrics]

        if not successful:
            return None

        if minimize:
            return min(successful, key=lambda r: r.metrics[metric])
        else:
            return max(successful, key=lambda r: r.metrics[metric])
