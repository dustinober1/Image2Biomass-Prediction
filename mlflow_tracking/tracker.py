"""
MLflow ExperimentTracker - Python SDK for systematic experiment logging.

This module provides a high-level interface to MLflow for tracking machine
learning experiments with automatic timestamp, status, and duration tracking.
"""

import time
from datetime import datetime
from typing import Optional

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities import Experiment

from mlflow_tracking.config import MLFLOW_TRACKING_URI
from mlflow_tracking.environment import get_environment, log_environment_to_mlflow


class ExperimentTracker:
    """
    A wrapper around MLflow for consistent experiment tracking.

    This class provides a simple Python SDK for logging experiments,
    including hyperparameters, metrics, and artifacts with automatic
    status and duration tracking.

    Attributes:
        client: MLflow client for tracking operations
        experiment: The MLflow experiment object
        active_run: The currently active MLflow run (if any)

    Example:
        >>> tracker = ExperimentTracker("my_experiment")
        >>> tracker.start_run("run_1", tags={"purpose": "test"})
        >>> tracker.log_params({"learning_rate": 0.01})
        >>> tracker.log_metrics({"rmse": 10.5})
        >>> tracker.end_run(status="completed")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        auto_log_environment: bool = True,
    ):
        """
        Initialize MLflow client and get or create experiment.

        Args:
            experiment_name: Name of the experiment to track
            tracking_uri: Optional MLflow tracking URI (defaults to config)
            auto_log_environment: If True, automatically log environment on run start
        """
        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        # Set the global MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        self.client: MlflowClient = MlflowClient(tracking_uri)
        self.experiment: Optional[Experiment] = self.client.get_experiment_by_name(
            experiment_name
        )

        if not self.experiment:
            # create_experiment returns experiment_id (string), need to fetch the Experiment object
            experiment_id = self.client.create_experiment(experiment_name)
            self.experiment = self.client.get_experiment(experiment_id)

        self.active_run: Optional[mlflow.active_run] = None
        self.auto_log_environment = auto_log_environment

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
        random_seed: Optional[int] = None,
    ) -> str:
        """
        Start a new MLflow run with automatic status and timestamp tracking.

        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to associate with the run
            random_seed: Optional random seed for reproducibility

        Returns:
            The unique run ID for the started run
        """
        # Set the active experiment before starting the run
        mlflow.set_experiment(self.experiment.name)

        # Start the run (uses the currently set experiment)
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)

        mlflow.set_tag("status", "running")
        mlflow.set_tag("start_time", datetime.now().isoformat())

        # Auto-log environment if enabled
        if self.auto_log_environment:
            log_environment_to_mlflow()

        # Log random seed if provided
        if random_seed is not None:
            mlflow.log_param("random_seed", random_seed)

        return self.active_run.info.run_id

    def log_params(self, params: dict) -> None:
        """
        Log hyperparameters or configuration values to the active run.

        Args:
            params: Dictionary of parameter names and values
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log evaluation metrics to the active run.

        Args:
            metrics: Dictionary of metric names and values (e.g., RMSE, R², MAE)
            step: Optional training step number for metric logging
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file to the active run.

        Args:
            file_path: Path to the artifact file (model, predictions CSV, etc.)
            artifact_path: Optional destination path within the artifact directory
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        mlflow.log_artifact(file_path, artifact_path)

    def log_environment(self, env: Optional[dict] = None) -> None:
        """
        Manually log environment metadata to the active run.

        Args:
            env: Optional environment dict. If None, captures current environment.
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        log_environment_to_mlflow(env)

    def add_tags(self, tags: dict) -> None:
        """
        Add tags to the currently active run.

        Tags are useful for organizing and filtering experiments, such as
        model_type, phase, or purpose. This is a convenience wrapper around
        mlflow.set_tags() for consistency with our SDK.

        Args:
            tags: Dictionary of tag key-value pairs

        Raises:
            RuntimeError: If no active run

        Example:
            >>> tracker = ExperimentTracker("my_experiment")
            >>> tracker.start_run("experiment_1")
            >>> tracker.add_tags({
            ...     "model_type": "random_forest",
            ...     "phase": "feature_ablation",
            ...     "purpose": "baseline"
            ... })
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        mlflow.set_tags(tags)

    def set_group(self, group_name: str, create_if_missing: bool = True) -> None:
        """
        Set the MLflow experiment (group) for the current run.

        Groups are logical containers for organizing related experiments.
        This method must be called BEFORE start_run() to take effect,
        as it sets the active experiment for subsequent runs.

        Args:
            group_name: Name of the experiment group
            create_if_missing: If True, creates the experiment if it doesn't exist

        Raises:
            ValueError: If group_name is empty

        Example:
            >>> tracker = ExperimentTracker("my_experiment")
            >>> tracker.set_group("ablation-studies")  # Must be before start_run()
            >>> tracker.start_run("experiment_1")
        """
        if not group_name or not isinstance(group_name, str):
            raise ValueError("group_name must be a non-empty string")

        if create_if_missing:
            # Ensure experiment exists
            experiment = self.client.get_experiment_by_name(group_name)
            if experiment is None:
                self.client.create_experiment(group_name)

        mlflow.set_experiment(group_name)

    def get_run_id(self) -> str:
        """
        Get the run ID of the currently active run.

        Useful for passing to ExperimentOrganizer methods or for
        manual run manipulation.

        Returns:
            The unique run ID of the active run

        Raises:
            RuntimeError: If no active run

        Example:
            >>> tracker = ExperimentTracker("my_experiment")
            >>> tracker.start_run("experiment_1")
            >>> run_id = tracker.get_run_id()
            >>> print(f"Current run: {run_id}")
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        return self.active_run.info.run_id

    def mark_failed(self, error_message: Optional[str] = None) -> None:
        """
        Mark the current run as failed with an optional error message.

        This is a convenience method for explicitly marking a run as failed,
        typically used when an adapter execution raises an exception.

        Args:
            error_message: Optional error message to log to MLflow

        Raises:
            RuntimeError: If no active run

        Example:
            >>> try:
            ...     adapter.execute(config, tracker)
            ... except Exception as e:
            ...     tracker.mark_failed(str(e))
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        if error_message:
            # Log error message as a tag for visibility in MLflow UI
            mlflow.set_tag("error_message", error_message)

        # End the run with failed status
        self.end_run(status="failed")

    def end_run(self, status: str = "completed") -> None:
        """
        End the active run with final status and duration.

        Args:
            status: Final status of the run ("completed", "failed", "killed")
        """
        if self.active_run:
            duration = time.time() - self.active_run.info.start_time / 1000
            mlflow.set_tag("status", status)
            mlflow.set_tag("duration", duration)
            mlflow.set_tag("end_time", datetime.now().isoformat())
            mlflow.end_run()
            self.active_run = None

    def __enter__(self):
        """
        Context manager entry point.

        Returns:
            self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - automatically end run with appropriate status.

        If an exception occurred during the context, marks the run as failed.
        Otherwise, marks the run as completed.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if exc_type is not None:
            self.end_run(status="failed")
        else:
            self.end_run(status="completed")
