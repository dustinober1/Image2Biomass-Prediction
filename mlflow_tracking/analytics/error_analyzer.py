"""
ErrorAnalyzer - Residual analysis and failure mode identification.

This module provides the ErrorAnalyzer class for analyzing prediction errors,
computing residual statistics, and identifying systematic failure patterns.
"""

from typing import Optional, Tuple, Dict
import warnings
import os
import tempfile

import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from scipy.stats import skew, kurtosis

from mlflow_tracking.config import MLFLOW_TRACKING_URI


class ErrorAnalyzer:
    """
    Analyze prediction errors and identify systematic failure modes.

    This class loads predictions from MLflow artifacts, computes residuals,
    generates diagnostic visualizations, and clusters high-error samples
    to identify failure patterns.

    Attributes:
        client: MLflow client for artifact loading
        tracking_uri: MLflow tracking URI
        predictions_df: DataFrame with columns [image_id, actual, predicted, residual, abs_residual, pct_error]
        run_id: ID of the currently loaded run

    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> analyzer.load_run("abc123", "predictions.csv")
        >>> residuals_df = analyzer.compute_residuals()
        >>> fig = analyzer.plot_residuals()
        >>> failure_modes = analyzer.identify_failure_modes(n_clusters=3)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client and analyzer state.

        Args:
            tracking_uri: Optional MLflow tracking URI (defaults to config)
        """
        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        self.client: MlflowClient = MlflowClient(tracking_uri)
        self.tracking_uri = tracking_uri
        self.predictions_df: Optional[pd.DataFrame] = None
        self.residuals: Optional[np.ndarray] = None
        self.run_id: Optional[str] = None

    def load_run(self, run_id: str, predictions_path: str = "predictions.csv") -> None:
        """
        Load predictions from MLflow run artifacts.

        Downloads the predictions CSV artifact from the specified run,
        loads it into a DataFrame, and computes residuals.

        Args:
            run_id: MLflow run ID to load predictions from
            predictions_path: Path to predictions CSV within artifact directory

        Raises:
            FileNotFoundError: If predictions artifact doesn't exist
            ValueError: If predictions CSV is missing required columns

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123", "predictions.csv")
            >>> print(analyzer.predictions_df.head())
        """
        # Fetch run info
        run = self.client.get_run(run_id)
        self.run_id = run_id

        # Download artifact to temporary directory
        artifact_uri = run.info.artifact_uri

        # Handle local file:// URIs
        if artifact_uri.startswith("file://"):
            artifact_path = artifact_uri.replace("file://", "")
            # Construct full path to predictions file
            predictions_file = os.path.join(artifact_path, predictions_path)
        else:
            # For non-local URIs, download to temp directory
            temp_dir = tempfile.mkdtemp()
            # download_artifacts returns the path to the downloaded directory
            # If predictions_path is a file, it will be downloaded as temp_dir/predictions_path
            self.client.download_artifacts(run_id, predictions_path, temp_dir)
            predictions_file = os.path.join(temp_dir, predictions_path)

        # Check if file exists
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(
                f"Predictions artifact not found: {predictions_file}. "
                f"Ensure predictions are logged as '{predictions_path}' in run {run_id}"
            )

        # Load predictions CSV
        self.predictions_df = pd.read_csv(predictions_file)

        # Validate required columns
        required_cols = ["image_id", "actual", "predicted"]
        missing_cols = [col for col in required_cols if col not in self.predictions_df.columns]
        if missing_cols:
            raise ValueError(
                f"Predictions CSV missing required columns: {missing_cols}. "
                f"Found columns: {list(self.predictions_df.columns)}"
            )

        # Compute residuals
        self.predictions_df["residual"] = self.predictions_df["actual"] - self.predictions_df["predicted"]
        self.predictions_df["abs_residual"] = self.predictions_df["residual"].abs()

        # Compute percentage error (handle division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictions_df["pct_error"] = (
                (self.predictions_df["residual"] / self.predictions_df["actual"]) * 100
            )

        # Store residuals as numpy array for statistical operations
        self.residuals = self.predictions_df["residual"].values

    def compute_residuals(self) -> pd.DataFrame:
        """
        Compute residual statistics for all predictions.

        Returns:
            DataFrame with columns:
            - image_id: Sample identifier
            - actual: Ground truth value
            - predicted: Model prediction
            - residual: actual - predicted
            - abs_residual: Absolute residual (error magnitude)
            - pct_error: Percentage error relative to actual

        Raises:
            RuntimeError: If no run has been loaded

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> residuals_df = analyzer.compute_residuals()
            >>> print(residuals_df.describe())
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        return self.predictions_df[[
            "image_id",
            "actual",
            "predicted",
            "residual",
            "abs_residual",
            "pct_error"
        ]].copy()

    def plot_residuals(self, figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
        """
        Create residual plot showing prediction errors vs predicted values.

        Uses a regression plot with locally weighted smoothing (LOWESS)
        to visualize trends in prediction errors across the prediction range.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            matplotlib Figure object for artifact logging

        Raises:
            RuntimeError: If no run has been loaded

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> fig = analyzer.plot_residuals()
            >>> fig.savefig("residuals.png")
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create residual plot with LOWESS smoothing
        sns.regplot(
            data=self.predictions_df,
            x="predicted",
            y="residual",
            lowess=True,
            line_kws={"color": "red", "lw": 2, "label": "LOWESS Trend"},
            scatter_kws={"alpha": 0.5},
            ax=ax
        )

        # Add reference line at y=0 (perfect prediction)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Zero Error")

        # Color residuals by error magnitude
        abs_residuals = self.predictions_df["abs_residual"].values
        normalize = plt.Normalize(vmin=abs_residuals.min(), vmax=abs_residuals.max())
        colors = plt.cm.coolwarm_r(normalize(abs_residuals))

        # Update scatter colors
        for i, point in enumerate(ax.collections[0].get_offsets()):
            ax.collections[0].get_paths()

        # Labels and title
        ax.set_xlabel("Predicted Values", fontsize=12)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
        ax.set_title("Residual Plot: Actual vs Predicted", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def plot_prediction_vs_actual(self, figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
        """
        Create prediction vs actual scatter plot with R² annotation.

        Shows how well predictions match ground truth values. Points along
        the diagonal line indicate perfect predictions.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            matplotlib Figure object for artifact logging

        Raises:
            RuntimeError: If no run has been loaded

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> fig = analyzer.plot_prediction_vs_actual()
            >>> fig.savefig("pred_vs_actual.png")
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot with color mapping by absolute error
        scatter = sns.scatterplot(
            data=self.predictions_df,
            x="actual",
            y="predicted",
            hue="abs_residual",
            palette="coolwarm_r",
            alpha=0.6,
            ax=ax
        )

        # Add diagonal reference line (perfect prediction)
        min_val = min(self.predictions_df["actual"].min(), self.predictions_df["predicted"].min())
        max_val = max(self.predictions_df["actual"].max(), self.predictions_df["predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Prediction")

        # Compute R²
        actual = self.predictions_df["actual"].values
        predicted = self.predictions_df["predicted"].values
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Add R² annotation
        ax.text(
            0.05, 0.95,
            f"R² = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        )

        # Labels and title
        ax.set_xlabel("Actual Biomass (g)", fontsize=12)
        ax.set_ylabel("Predicted Biomass (g)", fontsize=12)
        ax.set_title("Prediction vs Actual: Model Performance", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")

        plt.tight_layout()
        return fig

    def plot_error_distribution(self, figsize: Tuple[int, int] = (14, 5)) -> 'plt.Figure':
        """
        Create error distribution visualization with histogram and box plot.

        Shows the distribution of residuals to identify bias, skew, and outliers.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            matplotlib Figure object with 2 subplots

        Raises:
            RuntimeError: If no run has been loaded

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> fig = analyzer.plot_error_distribution()
            >>> fig.savefig("error_distribution.png")
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Subplot 1: Histogram with KDE
        residuals = self.predictions_df["residual"].values
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        sns.histplot(data=self.predictions_df, x="residual", kde=True, ax=axes[0])
        axes[0].axvline(mean_residual, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_residual:.2f}")

        # Add statistics text box
        textstr = f"Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axes[0].text(0.95, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment="top", horizontalalignment="right", bbox=props)

        axes[0].set_xlabel("Residual", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("Residual Distribution", fontsize=13, fontweight="bold")
        axes[0].legend()

        # Subplot 2: Box plot
        sns.boxplot(data=self.predictions_df, y="residual", ax=axes[1])
        sns.stripplot(data=self.predictions_df, y="residual", color="black", alpha=0.3, size=3, jitter=True, ax=axes[1])

        axes[1].set_ylabel("Residual", fontsize=12)
        axes[1].set_title("Error Distribution (Box Plot)", fontsize=13, fontweight="bold")

        plt.tight_layout()
        return fig

    def identify_failure_modes(
        self,
        n_clusters: int = 3,
        error_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Identify systematic failure patterns using K-means clustering.

        Clusters high-error samples to discover common failure modes
        (e.g., consistently underpredicting high biomass samples).

        Args:
            n_clusters: Number of clusters to identify
            error_threshold: Absolute error threshold for high-error samples.
                           If None, uses top 25% of errors.

        Returns:
            DataFrame with cluster assignments and statistics:
            - image_id: Sample identifier
            - actual: Ground truth value
            - predicted: Model prediction
            - residual: Prediction error
            - abs_residual: Absolute error
            - cluster: Cluster assignment (0 to n_clusters-1)
            - cluster_mean_abs_residual: Mean absolute error for cluster

        Raises:
            RuntimeError: If no run has been loaded
            ValueError: If fewer than 10 high-error samples (insufficient for clustering)

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> failure_modes = analyzer.identify_failure_modes(n_clusters=3)
            >>> print(failure_modes.groupby("cluster").size())
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        # Determine error threshold
        if error_threshold is None:
            # Use top 25% of errors
            error_threshold = self.predictions_df["abs_residual"].quantile(0.75)

        # Filter high-error samples
        high_error_mask = self.predictions_df["abs_residual"] > error_threshold
        high_error_df = self.predictions_df[high_error_mask].copy()

        # Validate sufficient samples for clustering
        if len(high_error_df) < 10:
            raise ValueError(
                f"Insufficient high-error samples for clustering: {len(high_error_df)}. "
                f"Need at least 10 samples. Consider lowering error_threshold."
            )

        # Extract features for clustering
        features = high_error_df[["actual", "predicted", "residual", "abs_residual"]].values

        # Perform K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Assign cluster labels
        high_error_df = high_error_df.copy()
        high_error_df["cluster"] = cluster_labels

        # Compute cluster statistics
        cluster_stats = high_error_df.groupby("cluster")["abs_residual"].mean().to_dict()

        # Add cluster mean statistics to DataFrame
        high_error_df["cluster_mean_abs_residual"] = high_error_df["cluster"].map(cluster_stats)

        return high_error_df

    def get_error_statistics(self) -> Dict[str, float]:
        """
        Compute comprehensive error statistics.

        Returns:
            Dictionary with error statistics:
            - mean_abs_error: Mean absolute error
            - median_abs_error: Median absolute error
            - max_abs_error: Maximum absolute error
            - std_residual: Standard deviation of residuals
            - p25, p50, p75, p90, p95, p99: Percentiles of absolute error
            - skewness: Skewness of residual distribution
            - kurtosis: Kurtosis of residual distribution

        Raises:
            RuntimeError: If no run has been loaded

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> analyzer.load_run("abc123")
            >>> stats = analyzer.get_error_statistics()
            >>> print(f"MAE: {stats['mean_abs_error']:.2f}")
        """
        if self.predictions_df is None:
            raise RuntimeError("No predictions loaded. Call load_run() first.")

        abs_residuals = self.predictions_df["abs_residual"].values
        residuals = self.predictions_df["residual"].values

        # Compute percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        p_values = np.percentile(abs_residuals, percentiles)

        # Build statistics dictionary
        stats = {
            "mean_abs_error": float(np.mean(abs_residuals)),
            "median_abs_error": float(np.median(abs_residuals)),
            "max_abs_error": float(np.max(abs_residuals)),
            "std_residual": float(np.std(residuals)),
            "p25": float(p_values[0]),
            "p50": float(p_values[1]),
            "p75": float(p_values[2]),
            "p90": float(p_values[3]),
            "p95": float(p_values[4]),
            "p99": float(p_values[5]),
            "skewness": float(skew(residuals)),
            "kurtosis": float(kurtosis(residuals)),
        }

        return stats
