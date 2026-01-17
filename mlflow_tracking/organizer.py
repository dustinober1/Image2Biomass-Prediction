"""
MLflow ExperimentOrganizer - Python SDK for experiment organization and discovery.

This module provides a high-level interface to MLflow for organizing experiments
through logical groups, descriptive tags, and powerful search capabilities.
"""

from typing import Optional, List, Dict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment

from mlflow_tracking.config import MLFLOW_TRACKING_URI


class ExperimentOrganizer:
    """
    A wrapper around MLflow for organizing and discovering experiments.

    This class provides methods for creating experiment groups, tagging runs,
    searching experiments by metrics/parameters/tags, and finding best runs.

    Attributes:
        client: MLflow client for tracking operations

    Example:
        >>> organizer = ExperimentOrganizer()
        >>> group_id = organizer.create_group("ablation-studies", {"purpose": "feature_importance"})
        >>> organizer.add_tags_to_run(run_id, {"model_type": "random_forest"})
        >>> runs = organizer.search_runs(filter_string="tags.model_type = 'random_forest'")
        >>> best = organizer.get_best_runs("ablation-studies", "val_rmse", top_k=3)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client for organization operations.

        Args:
            tracking_uri: Optional MLflow tracking URI (defaults to config)
        """
        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(tracking_uri)
        self.client: MlflowClient = MlflowClient(tracking_uri)

    def create_group(self, group_name: str, tags: Optional[dict] = None) -> str:
        """
        Create an experiment group (MLflow experiment) for organizing related runs.

        Groups are logical containers for experiments, such as "ablation-studies",
        "ensemble-tests", or "hyperparameter-tuning". Uses MLflow's built-in
        experiment mechanism.

        Args:
            group_name: Name of the experiment group to create
            tags: Optional dictionary of tags to describe the group
                  (e.g., {"purpose": "ablation-study", "project": "biomass"})

        Returns:
            The experiment_id of the created or existing group

        Raises:
            ValueError: If group_name is empty or None

        Example:
            >>> organizer = ExperimentOrganizer()
            >>> exp_id = organizer.create_group("ablation-studies")
            >>> exp_id = organizer.create_group("ensemble-tests",
            ...                                 tags={"purpose": "ensemble_methods"})
        """
        if not group_name or not isinstance(group_name, str):
            raise ValueError("group_name must be a non-empty string")

        # Check if experiment already exists
        experiment = self.client.get_experiment_by_name(group_name)

        if experiment is None:
            # Create new experiment
            experiment_id = self.client.create_experiment(group_name, tags=tags)
        else:
            # Experiment exists - update tags if provided
            experiment_id = experiment.experiment_id
            if tags:
                # Update tags on existing experiment
                for key, value in tags.items():
                    self.client.set_experiment_tag(experiment_id, key, value)

        return experiment_id

    def add_tags_to_run(self, run_id: str, tags: dict) -> None:
        """
        Add tags to an existing run for post-hoc organization.

        Useful for marking successful runs, adding model type information,
        or categorizing experiments after they've completed.

        Args:
            run_id: The unique run identifier
            tags: Dictionary of tag key-value pairs to add

        Raises:
            ValueError: If run_id is empty or tags is empty
            mlflow.exceptions.MlflowException: If run_id doesn't exist

        Example:
            >>> organizer = ExperimentOrganizer()
            >>> organizer.add_tags_to_run(run_id, {
            ...     "model_type": "random_forest",
            ...     "purpose": "baseline",
            ...     "status": "completed"
            ... })
        """
        if not run_id or not isinstance(run_id, str):
            raise ValueError("run_id must be a non-empty string")

        if not tags or not isinstance(tags, dict):
            raise ValueError("tags must be a non-empty dictionary")

        # Add each tag to the run
        for key, value in tags.items():
            self.client.set_tag(run_id, key, value)

    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: str = "",
        max_results: int = 1000,
        order_by: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for runs using MLflow's powerful query language.

        Supports filtering by metrics, parameters, and tags using MLflow's
        filter string syntax.

        Args:
            experiment_ids: Optional list of experiment IDs to search within.
                           If None, searches all experiments.
            filter_string: MLflow filter string for querying runs.
                          Examples:
                          - "metrics.val_rmse < 10.0"
                          - "params.model_type = 'random_forest'"
                          - "tags.status = 'completed'"
                          - "params.model_type = 'xgboost' and metrics.test_rmse < 15.0"
            max_results: Maximum number of results to return (default: 1000)
            order_by: Optional list of order by clauses.
                     Examples: ["metrics.val_rmse ASC"], ["params.n_estimators DESC"]

        Returns:
            List of simplified run dictionaries containing:
            - run_id: Unique run identifier
            - experiment_id: Experiment containing the run
            - params: Dictionary of hyperparameters
            - metrics: Dictionary of metrics
            - tags: Dictionary of tags
            - start_time: Start timestamp (milliseconds since epoch)
            - status: Run status ("RUNNING", "COMPLETED", "FAILED", "KILLED")

        Raises:
            ValueError: If max_results is not positive

        Example:
            >>> organizer = ExperimentOrganizer()
            >>> # Find all completed runs with low validation error
            >>> runs = organizer.search_runs(
            ...     filter_string="tags.status = 'completed' and metrics.val_rmse < 10.0"
            ... )
            >>> # Find XGBoost runs sorted by validation RMSE
            >>> runs = organizer.search_runs(
            ...     filter_string="params.model_type = 'xgboost'",
            ...     order_by=["metrics.val_rmse ASC"]
            ... )
            >>> for run in runs:
            ...     print(f"Run {run['run_id']}: RMSE={run['metrics']['val_rmse']:.2f}")
        """
        if max_results <= 0:
            raise ValueError("max_results must be a positive integer")

        if order_by is None:
            order_by = []

        # Use mlflow.search_runs directly
        runs_df = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            output_format="pandas"
        )

        # Convert DataFrame to list of simplified dictionaries
        results = []
        for _, row in runs_df.iterrows():
            result = {
                "run_id": row.get("run_id", ""),
                "experiment_id": str(row.get("experiment_id", "")),
                "params": {},
                "metrics": {},
                "tags": {},
                "start_time": row.get("start_time", 0),
                "status": row.get("status", "UNKNOWN")
            }

            # Extract params, metrics, tags (column names vary by MLflow version)
            for col in runs_df.columns:
                if col.startswith("params."):
                    param_name = col.replace("params.", "")
                    result["params"][param_name] = row[col]
                elif col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    result["metrics"][metric_name] = row[col]
                elif col.startswith("tags."):
                    tag_name = col.replace("tags.", "")
                    result["tags"][tag_name] = row[col]

            results.append(result)

        return results

    def list_groups(self) -> List[Dict]:
        """
        List all available experiment groups.

        Returns all experiments (groups) with their metadata, useful for
        discovering what groups are available for searching or organizing.

        Returns:
            List of experiment dictionaries containing:
            - experiment_id: Unique experiment identifier
            - name: Experiment name
            - artifact_location: Path to artifact storage
            - tags: Dictionary of experiment-level tags
            - creation_time: Creation timestamp (milliseconds since epoch)

        Example:
            >>> organizer = ExperimentOrganizer()
            >>> groups = organizer.list_groups()
            >>> for group in groups:
            ...     print(f"{group['name']}: {group.get('tags', {})}")
        """
        experiments = self.client.search_experiments()

        results = []
        for exp in experiments:
            result = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "tags": exp.tags if exp.tags else {},
                "creation_time": exp.creation_time if hasattr(exp, 'creation_time') else 0
            }
            results.append(result)

        return results

    def get_best_runs(
        self,
        group_name: str,
        metric_name: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find the top K runs in a group by specified metric.

        Searches within a specific experiment group and returns runs sorted
        by the specified metric in ascending order (lower is better).

        Args:
            group_name: Name of the experiment group to search
            metric_name: Name of the metric to optimize (e.g., "val_rmse")
            top_k: Number of top runs to return (default: 5)

        Returns:
            List of run dictionaries sorted by metric (best first)

        Raises:
            ValueError: If group_name or metric_name is empty, or top_k is not positive

        Example:
            >>> organizer = ExperimentOrganizer()
            >>> best_runs = organizer.get_best_runs("ablation-studies", "val_rmse", top_k=3)
            >>> for i, run in enumerate(best_runs, 1):
            ...     print(f"#{i}: Run {run['run_id']} - RMSE={run['metrics']['val_rmse']:.2f}")
        """
        if not group_name or not isinstance(group_name, str):
            raise ValueError("group_name must be a non-empty string")

        if not metric_name or not isinstance(metric_name, str):
            raise ValueError("metric_name must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Get experiment by name
        experiment = self.client.get_experiment_by_name(group_name)
        if experiment is None:
            return []

        # Search runs sorted by metric (ascending = lower is better)
        runs = self.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"],
            max_results=top_k
        )

        return runs


def create_group(group_name: str, tags: Optional[dict] = None) -> str:
    """
    Convenience function to create an experiment group.

    Args:
        group_name: Name of the experiment group to create
        tags: Optional dictionary of tags to describe the group

    Returns:
        The experiment_id of the created or existing group

    Example:
        >>> from mlflow_tracking import create_group
        >>> exp_id = create_group("hyperparameter-tuning", tags={"project": "biomass"})
    """
    organizer = ExperimentOrganizer()
    return organizer.create_group(group_name, tags)
