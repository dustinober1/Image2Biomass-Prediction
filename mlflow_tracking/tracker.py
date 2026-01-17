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

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client and get or create experiment.

        Args:
            experiment_name: Name of the experiment to track
            tracking_uri: Optional MLflow tracking URI (defaults to config)
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

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None) -> str:
        """
        Start a new MLflow run with automatic status and timestamp tracking.

        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to associate with the run

        Returns:
            The unique run ID for the started run
        """
        # Set the active experiment before starting the run
        mlflow.set_experiment(self.experiment.name)

        # Start the run (uses the currently set experiment)
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)

        mlflow.set_tag("status", "running")
        mlflow.set_tag("start_time", datetime.now().isoformat())
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
