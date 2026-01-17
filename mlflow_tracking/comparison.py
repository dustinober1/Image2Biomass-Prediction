"""
MLflow ExperimentComparator - Python SDK for experiment comparison and analysis.

This module provides a high-level interface to MLflow for comparing multiple experiments,
aggregating results, and generating insights through clustering, correlation, and outlier detection.
"""

from typing import Optional, List, Dict, Union, Any
import warnings

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment

from mlflow_tracking.config import MLFLOW_TRACKING_URI


class ExperimentComparator:
    """
    A wrapper around MLflow for comparing and analyzing experiments.

    This class provides methods for comparing runs by IDs, groups, or filters,
    exporting results to various formats, and generating insights through
    clustering, correlation, and outlier detection.

    Attributes:
        client: MLflow client for tracking operations

    Example:
        >>> comparator = ExperimentComparator()
        >>> df = comparator.compare_by_ids(["run1", "run2", "run3"])
        >>> correlations = comparator.correlate_params(df)
        >>> outliers = comparator.find_outliers(df)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client for comparison operations.

        Args:
            tracking_uri: Optional MLflow tracking URI (defaults to config)
        """
        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(tracking_uri)
        self.client: MlflowClient = MlflowClient(tracking_uri)

    def compare_by_ids(
        self,
        run_ids: List[str],
        as_dataframe: bool = True,
        required_metrics: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Compare runs by explicit run IDs.

        Fetches each run individually using get_run() and returns a structured
        comparison of their parameters, metrics, tags, and metadata.

        Args:
            run_ids: List of run IDs to compare
            as_dataframe: If True, return DataFrame; otherwise return list of dicts
            required_metrics: Optional list of metrics that must be present (validation skipped if None)

        Returns:
            DataFrame or list of dicts containing:
            - run_id: Unique run identifier
            - experiment_id: Experiment containing the run
            - params: Hyperparameters (columns prefixed with 'params.')
            - metrics: Evaluation metrics (columns prefixed with 'metrics.')
            - tags: Tags (columns prefixed with 'tags.')
            - start_time: Start timestamp
            - status: Run status

        Raises:
            ValueError: If run_ids is empty or contains invalid IDs

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_ids(["abc123", "def456"])
            >>> print(df[["run_id", "metrics.val_rmse", "params.n_estimators"]])
        """
        if not run_ids or not isinstance(run_ids, list):
            raise ValueError("run_ids must be a non-empty list of run IDs")

        results = []

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)

                result = {
                    "run_id": run.info.run_id,
                    "experiment_id": str(run.info.experiment_id),
                    "start_time": run.info.start_time,
                    "status": run.info.status,
                }

                # Extract parameters
                for key, value in run.data.params.items():
                    result[f"params.{key}"] = value

                # Extract metrics
                for key, value in run.data.metrics.items():
                    result[f"metrics.{key}"] = value

                # Extract tags
                for key, value in run.data.tags.items():
                    result[f"tags.{key}"] = value

                results.append(result)

            except Exception as e:
                warnings.warn(f"Failed to fetch run {run_id}: {e}")
                # Add placeholder for failed runs
                results.append({
                    "run_id": run_id,
                    "experiment_id": None,
                    "start_time": None,
                    "status": "FAILED",
                })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Auto-sort by primary metric if available
        if not df.empty:
            primary_metric = self._get_primary_metric(df.columns)
            if primary_metric and primary_metric in df.columns:
                df = df.sort_values(by=primary_metric, ascending=True)

        # Validate required metrics
        if required_metrics:
            self.validate_required_metrics(df, required_metrics)

        if as_dataframe:
            return df.reset_index(drop=True)
        else:
            return df.to_dict("records")

    def compare_by_group(
        self,
        group_name: str,
        as_dataframe: bool = True,
        required_metrics: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Compare all runs in an experiment group.

        Uses MLflow's experiment mechanism to fetch all runs within a
        specific experiment group.

        Args:
            group_name: Name of the experiment group
            as_dataframe: If True, return DataFrame; otherwise return list of dicts
            required_metrics: Optional list of metrics that must be present

        Returns:
            DataFrame or list of dicts (same format as compare_by_ids)
            Returns empty DataFrame/dict if group doesn't exist

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_group("ablation-studies")
            >>> print(df[["run_id", "metrics.val_rmse"]])
        """
        # Get experiment by name
        experiment = self.client.get_experiment_by_name(group_name)

        if experiment is None:
            # Return empty result
            if as_dataframe:
                return pd.DataFrame()
            else:
                return []

        # Search all runs in the experiment
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            output_format="pandas"
        )

        # Convert to our standard format
        results = []
        for _, row in runs_df.iterrows():
            result = {
                "run_id": row.get("run_id", ""),
                "experiment_id": str(row.get("experiment_id", "")),
                "start_time": row.get("start_time", 0),
                "status": row.get("status", "UNKNOWN"),
            }

            # Extract params, metrics, tags
            for col in runs_df.columns:
                if col.startswith("params."):
                    result[col] = row[col]
                elif col.startswith("metrics."):
                    result[col] = row[col]
                elif col.startswith("tags."):
                    result[col] = row[col]

            results.append(result)

        df = pd.DataFrame(results)

        # Auto-sort by primary metric
        if not df.empty:
            primary_metric = self._get_primary_metric(df.columns)
            if primary_metric and primary_metric in df.columns:
                df = df.sort_values(by=primary_metric, ascending=True)

        # Validate required metrics
        if required_metrics and not df.empty:
            self.validate_required_metrics(df, required_metrics)

        if as_dataframe:
            return df.reset_index(drop=True)
        else:
            return df.to_dict("records")

    def compare_by_filter(
        self,
        filter_string: str,
        max_results: int = 100,
        as_dataframe: bool = True,
        required_metrics: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Compare runs matching MLflow filter expression.

        Uses MLflow's powerful filter syntax to search runs by metrics,
        parameters, and tags.

        Args:
            filter_string: MLflow filter expression
                Examples: "metrics.val_rmse < 10.0"
                          "params.model_type = 'random_forest'"
                          "params.n_estimators >= 100 and metrics.test_rmse < 15.0"
            max_results: Maximum number of results to return
            as_dataframe: If True, return DataFrame; otherwise return list of dicts
            required_metrics: Optional list of metrics that must be present

        Returns:
            DataFrame or list of dicts (same format as compare_by_ids)

        Raises:
            ValueError: If filter_string is empty

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_filter("metrics.val_rmse < 10.0")
            >>> print(df[["run_id", "metrics.val_rmse", "params.model_type"]])
        """
        if not filter_string or not isinstance(filter_string, str):
            raise ValueError("filter_string must be a non-empty string")

        # Search runs using filter
        runs_df = mlflow.search_runs(
            filter_string=filter_string,
            max_results=max_results,
            output_format="pandas"
        )

        # Convert to our standard format
        results = []
        for _, row in runs_df.iterrows():
            result = {
                "run_id": row.get("run_id", ""),
                "experiment_id": str(row.get("experiment_id", "")),
                "start_time": row.get("start_time", 0),
                "status": row.get("status", "UNKNOWN"),
            }

            # Extract params, metrics, tags
            for col in runs_df.columns:
                if col.startswith("params."):
                    result[col] = row[col]
                elif col.startswith("metrics."):
                    result[col] = row[col]
                elif col.startswith("tags."):
                    result[col] = row[col]

            results.append(result)

        df = pd.DataFrame(results)

        # Auto-sort by primary metric
        if not df.empty:
            primary_metric = self._get_primary_metric(df.columns)
            if primary_metric and primary_metric in df.columns:
                df = df.sort_values(by=primary_metric, ascending=True)

        # Validate required metrics
        if required_metrics and not df.empty:
            self.validate_required_metrics(df, required_metrics)

        if as_dataframe:
            return df.reset_index(drop=True)
        else:
            return df.to_dict("records")

    def validate_required_metrics(
        self,
        df: pd.DataFrame,
        required_metrics: List[str]
    ) -> None:
        """
        Validate that required metrics exist in the DataFrame.

        Checks each metric in required_metrics exists in DataFrame columns.
        Failed runs (status="FAILED") are excluded from validation.

        Args:
            df: DataFrame from compare_by_* methods
            required_metrics: List of metric names that must be present

        Raises:
            ValueError: If any required metrics are missing

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_ids(["run1"])
            >>> comparator.validate_required_metrics(df, ["train.rmse", "val.rmse"])
        """
        if df.empty:
            return

        # Exclude failed runs from validation
        if "status" in df.columns:
            df_to_check = df[df["status"] != "FAILED"]
        else:
            df_to_check = df

        if df_to_check.empty:
            return

        # Check for missing metrics
        missing_metrics = []
        for metric in required_metrics:
            metric_col = f"metrics.{metric}"
            if metric_col not in df.columns:
                missing_metrics.append(metric)

        if missing_metrics:
            raise ValueError(
                f"Required metrics missing from DataFrame: {missing_metrics}. "
                f"Available metric columns: {[c for c in df.columns if c.startswith('metrics.')]}"
            )

    def to_csv(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Export comparison results to CSV format.

        Args:
            df: DataFrame from compare_by_* methods
            filepath: Path to output CSV file

        Returns:
            filepath on success

        Raises:
            ValueError: If filepath is empty

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_ids(["run1", "run2"])
            >>> comparator.to_csv(df, "results/comparison.csv")
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")

        df.to_csv(filepath, index=False)
        return filepath

    def to_json(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Export comparison results to JSON format.

        Args:
            df: DataFrame from compare_by_* methods
            filepath: Path to output JSON file

        Returns:
            filepath on success

        Raises:
            ValueError: If filepath is empty

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_ids(["run1", "run2"])
            >>> comparator.to_json(df, "results/comparison.json")
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")

        df.to_json(filepath, orient="records", indent=2)
        return filepath

    def to_excel(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Export comparison results to Excel format.

        Args:
            df: DataFrame from compare_by_* methods
            filepath: Path to output Excel file

        Returns:
            filepath on success

        Raises:
            ValueError: If filepath is empty or openpyxl not installed

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_ids(["run1", "run2"])
            >>> comparator.to_excel(df, "results/comparison.xlsx")
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df.to_excel(filepath, index=False, engine="openpyxl")
        except ImportError:
            raise ValueError(
                "openpyxl package is required for Excel export. "
                "Install it with: pip install openpyxl"
            )

        return filepath

    def cluster_runs(
        self,
        runs_df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        features: Union[List[str], str] = "auto"
    ) -> Dict[str, Any]:
        """
        Perform K-means clustering on experiment results.

        Groups similar experiments based on their metrics to identify patterns
        and configurations that lead to similar performance.

        Args:
            runs_df: DataFrame from compare_by_* methods
            n_clusters: Number of clusters (uses elbow heuristic if None)
            features: Metric columns to use for clustering
                - "auto": All metric columns (default)
                - List of column names: Specific metrics

        Returns:
            Dictionary with:
            - cluster_labels: Array of cluster assignments for each run
            - cluster_centers: Array of cluster centers (metric values)
            - inertia: Within-cluster sum of squares
            - n_clusters: Number of clusters used

        Raises:
            ValueError: If fewer than 2 runs or insufficient metric columns

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_group("hyperparameter-tuning")
            >>> clusters = comparator.cluster_runs(df, n_clusters=3)
            >>> print(f"Found {clusters['n_clusters']} clusters")
        """
        if len(runs_df) < 2:
            raise ValueError("At least 2 runs are required for clustering")

        # Extract metric columns
        if features == "auto":
            metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
        else:
            metric_cols = features

        if not metric_cols:
            raise ValueError("No metric columns found for clustering")

        # Extract metric values, handle NaN
        X = runs_df[metric_cols].copy()
        X = X.fillna(X.mean())  # Fill NaN with column mean

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(5, len(runs_df) // 2)
            if n_clusters < 2:
                n_clusters = 2

        # Import sklearn
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install it with: pip install scikit-learn"
            )

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        return {
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
            "n_clusters": n_clusters,
        }

    def correlate_params(
        self,
        runs_df: pd.DataFrame,
        method: str = "pearson",
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute correlations between parameters and metrics.

        Identifies which hyperparameters are most strongly associated with
        performance metrics.

        Args:
            runs_df: DataFrame from compare_by_* methods
            method: Correlation method ("pearson" or "spearman")
            threshold: Minimum absolute correlation to include in results

        Returns:
            DataFrame with columns:
            - param: Parameter name
            - metric: Metric name
            - correlation: Correlation coefficient

        Raises:
            ValueError: If method is not "pearson" or "spearman"

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_group("hyperparameter-tuning")
            >>> corr = comparator.correlate_params(df, threshold=0.3)
            >>> print(corr.sort_values("correlation", ascending=False))
        """
        if method not in ["pearson", "spearman"]:
            raise ValueError('method must be "pearson" or "spearman"')

        # Extract numeric params and metrics
        param_cols = [c for c in runs_df.columns if c.startswith("params.")]
        metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]

        # Build correlation matrix
        correlations = []

        for param_col in param_cols:
            # Try to convert param to numeric
            param_series = pd.to_numeric(runs_df[param_col], errors="coerce")

            for metric_col in metric_cols:
                metric_series = runs_df[metric_col]

                # Drop NaN values
                valid_idx = ~(param_series.isna() | metric_series.isna())

                if valid_idx.sum() < 2:
                    continue

                # Compute correlation
                corr = param_series[valid_idx].corr(metric_series[valid_idx], method=method)

                if not pd.isna(corr) and abs(corr) >= threshold:
                    correlations.append({
                        "param": param_col.replace("params.", ""),
                        "metric": metric_col.replace("metrics.", ""),
                        "correlation": corr
                    })

        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)

        if not corr_df.empty:
            corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)
            corr_df = corr_df.reset_index(drop=True)

        return corr_df

    def find_outliers(
        self,
        runs_df: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Identify outlier experiments based on metrics.

        Detects anomalous runs that perform unusually well or poorly
        compared to the distribution.

        Args:
            runs_df: DataFrame from compare_by_* methods
            method: Outlier detection method
                - "zscore": Flag |z| > threshold (default)
                - "iqr": Flag outside Q1 - 1.5*IQR, Q3 + 1.5*IQR
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with:
            - outlier_runs: List of run_ids flagged as outliers
            - outlier_scores: Dict of run_id -> outlier score
            - method: Method used
            - threshold: Threshold used

        Example:
            >>> comparator = ExperimentComparator()
            >>> df = comparator.compare_by_group("hyperparameter-tuning")
            >>> outliers = comparator.find_outliers(df, method="zscore", threshold=2.5)
            >>> print(f"Found {len(outliers['outlier_runs'])} outliers")
        """
        # Extract metric columns
        metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]

        if not metric_cols:
            return {
                "outlier_runs": [],
                "outlier_scores": {},
                "method": method,
                "threshold": threshold,
            }

        # Get run IDs
        run_ids = runs_df["run_id"].tolist()

        # Extract metric values, drop NaN
        X = runs_df[metric_cols].copy()

        # Compute outlier scores
        outlier_scores = {}

        if method == "zscore":
            try:
                from scipy.stats import zscore
            except ImportError:
                # Fallback: manual z-score computation
                def zscore(arr):
                    mean = sum(arr) / len(arr)
                    std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5
                    if std == 0:
                        return [0] * len(arr)
                    return [(x - mean) / std for x in arr]

            # Compute z-scores for each metric
            z_scores = X.apply(lambda col: zscore(col.dropna()))
            z_scores = z_scores.fillna(0)

            # Flag outliers (any metric exceeds threshold)
            for i, run_id in enumerate(run_ids):
                max_z = max(abs(z_scores.iloc[i])) if i < len(z_scores) else 0
                outlier_scores[run_id] = float(max_z)

        elif method == "iqr":
            # Compute IQR bounds for each metric
            outlier_flags = []

            for col in metric_cols:
                values = X[col].dropna()
                if len(values) == 0:
                    continue

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                # Flag outliers
                is_outlier = (X[col] < lower_bound) | (X[col] > upper_bound)
                outlier_flags.append(is_outlier)

            # Combine flags across metrics
            if outlier_flags:
                combined_flags = pd.DataFrame(outlier_flags).T.any(axis=1)
                for i, run_id in enumerate(run_ids):
                    outlier_scores[run_id] = 1.0 if combined_flags.iloc[i] else 0.0

        else:
            raise ValueError(f'Unknown method: {method}. Use "zscore" or "iqr"')

        # Identify outlier runs
        outlier_runs = [run_id for run_id, score in outlier_scores.items() if score > threshold]

        return {
            "outlier_runs": outlier_runs,
            "outlier_scores": outlier_scores,
            "method": method,
            "threshold": threshold,
        }

    def _get_primary_metric(self, columns: List[str]) -> Optional[str]:
        """
        Determine the primary metric for sorting.

        Looks for common validation metrics in order of preference.

        Args:
            columns: List of DataFrame column names

        Returns:
            Column name of primary metric, or None if not found
        """
        preferred_metrics = [
            "metrics.val.rmse",
            "metrics.val_rmse",
            "metrics.val.r2",
            "metrics.val_r2",
            "metrics.val.rmse",
            "metrics.val_loss",
            "metrics.val_loss",
            "metrics.test.rmse",
            "metrics.test_rmse",
        ]

        for metric in preferred_metrics:
            if metric in columns:
                return metric

        # Fallback: first metric column
        metric_cols = [c for c in columns if c.startswith("metrics.") and c != "metrics.status"]
        if metric_cols:
            return metric_cols[0]

        return None
