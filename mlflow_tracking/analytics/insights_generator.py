"""
MLflow InsightsGenerator - Automated insights generation from experiment comparisons.

This module provides statistical testing, effect size calculation, and automated
recommendations for experiment comparisons.
"""

from typing import Optional, List, Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from mlflow_tracking.comparison import ExperimentComparator


class InsightsGenerator:
    """
    Generate automated insights from experiment comparisons.

    This class provides methods for statistical significance testing, effect size
    calculation, hyperparameter correlation analysis, and automated recommendations
    based on experiment results.

    Attributes:
        client: MLflow client for tracking operations
        comparator: ExperimentComparator instance for data fetching

    Example:
        >>> generator = InsightsGenerator()
        >>> insights = generator.generate_insights(run_ids, metric="val.rmse")
        >>> print(insights["recommendation"])
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client and ExperimentComparator instance.

        Args:
            tracking_uri: Optional MLflow tracking URI (defaults to config)
        """
        from mlflow_tracking.config import MLFLOW_TRACKING_URI

        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        self.comparator = ExperimentComparator(tracking_uri=tracking_uri)

    def _check_normality(
        self, data: np.ndarray, alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check if data is normally distributed using Shapiro-Wilk test.

        Args:
            data: Array of values to test for normality
            alpha: Significance level (default: 0.05)

        Returns:
            Tuple of (is_normal, p_value) where is_normal is True if p_value > alpha
        """
        if len(data) < 20:
            warnings.warn(
                f"Small sample size (n={len(data)}) for normality test. "
                "Consider using Mann-Whitney U test instead."
            )

        try:
            statistic, p_value = stats.shapiro(data)
            is_normal = p_value > alpha
            return is_normal, p_value
        except Exception as e:
            warnings.warn(f"Shapiro-Wilk test failed: {e}. Assuming non-normal distribution.")
            return False, 0.0

    def perform_statistical_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        test_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test between two groups.

        Automatically selects t-test or Mann-Whitney U based on normality,
        or allows manual test selection.

        Args:
            group1: First group of values
            group2: Second group of values
            alpha: Significance level (default: 0.05)
            test_type: Type of test to perform
                - "auto": Automatically select based on normality (default)
                - "ttest": Use t-test regardless of normality
                - "mannwhitney": Use Mann-Whitney U regardless of normality

        Returns:
            Dictionary with:
                - test_type: str (name of test used)
                - statistic: float (test statistic)
                - p_value: float (significance p-value)
                - significant: bool (p_value < alpha)
                - effect_size: float (Cohen's d)
                - effect_size_interpretation: str (small/medium/large)
                - mean_group1: float
                - mean_group2: float
                - improvement_pct: float ((mean_group2 - mean_group1) / mean_group1 * 100)

        Raises:
            ValueError: If test_type is invalid
        """
        # Calculate means
        mean1 = float(np.mean(group1))
        mean2 = float(np.mean(group2))

        # Calculate effect size
        effect_size = self._calculate_effect_size(group1, group2)
        effect_size_interpretation = self._interpret_effect_size(effect_size)

        # Calculate improvement percentage
        improvement_pct = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0.0

        # Select and perform test
        if test_type == "auto":
            # Check normality of both groups
            is_normal1, _ = self._check_normality(group1, alpha=alpha)
            is_normal2, _ = self._check_normality(group2, alpha=alpha)

            if is_normal1 and is_normal2:
                test_type = "ttest"
            else:
                test_type = "mannwhitney"

        if test_type == "ttest":
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_name = "t-test"
        elif test_type == "mannwhitney":
            statistic, p_value = stats.mannwhitneyu(
                group1, group2, alternative="two-sided"
            )
            test_name = "Mann-Whitney U"
        else:
            raise ValueError(
                f'Invalid test_type: {test_type}. Use "auto", "ttest", or "mannwhitney".'
            )

        return {
            "test_type": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "effect_size": effect_size,
            "effect_size_interpretation": effect_size_interpretation,
            "mean_group1": mean1,
            "mean_group2": mean2,
            "improvement_pct": improvement_pct,
        }

    def _calculate_effect_size(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size (0 if groups are identical)
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Compute pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Handle edge case: identical groups
        if pooled_std == 0:
            return 0.0

        # Compute Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        return float(cohens_d)

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Uses Cohen (1988) conventions:
        - < 0.2: negligible
        - 0.2 - 0.5: small
        - 0.5 - 0.8: medium
        - >= 0.8: large

        Args:
            cohens_d: Cohen's d effect size

        Returns:
            Interpretation string (negligible/small/medium/large)
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_recommendation(
        self,
        p_value: float,
        effect_size: float,
        improvement_pct: float,
        alpha: float = 0.05
    ) -> str:
        """
        Generate actionable recommendation from statistical results.

        Args:
            p_value: Statistical significance p-value
            effect_size: Cohen's d effect size
            improvement_pct: Percentage improvement
            alpha: Significance level

        Returns:
            Actionable recommendation string
        """
        if p_value >= alpha:
            return "No significant difference detected. Collect more data or refine experiment design."

        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return "Significant but negligible effect size. Consider practical significance."
        elif abs_effect < 0.5:
            return "Significant small effect. Consider cost-benefit trade-offs."
        elif abs_effect < 0.8:
            return "Significant medium effect. Recommended for deployment."
        else:
            return f"Significant large effect ({improvement_pct:.1f}% improvement). Strongly recommended for deployment."

    def generate_insights(
        self,
        run_ids: List[str],
        metric: str = "val.rmse",
        group_by: Optional[str] = None,
        alpha: float = 0.05,
        min_sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Generate automated insights from multiple runs.

        Performs statistical analysis and generates actionable recommendations
        based on experiment results.

        Args:
            run_ids: List of run IDs to analyze
            metric: Metric name to analyze (default: "val.rmse")
            group_by: Optional parameter to group by (e.g., "params.learning_rate")
            alpha: Significance level (default: 0.05)
            min_sample_size: Minimum runs required for analysis (default: 5)

        Returns:
            Dictionary with:
                - status: str ("success" or "insufficient_data")
                - best_run: str (run_id)
                - best_metric: float
                - summary_statistics: dict (mean, std, min, max, range)
                - statistical_tests: list of dicts (one per comparison)
                - overall_significant: bool (any test significant)
                - recommendation: str (actionable recommendation)
                - sample_size: int (number of runs)
                - metric: str (metric analyzed)
        """
        # Validate sample size
        if len(run_ids) < min_sample_size:
            return {
                "status": "insufficient_data",
                "recommendation": f"Insufficient data: {len(run_ids)} runs < {min_sample_size} minimum. Run more experiments to generate insights.",
                "sample_size": len(run_ids),
                "metric": metric,
            }

        # Fetch run data
        df = self.comparator.compare_by_ids(run_ids, as_dataframe=True)

        # Extract metric values
        metric_col = f"metrics.{metric}"

        if metric_col not in df.columns:
            raise ValueError(
                f"Metric {metric} not found in runs. "
                f"Available metrics: {[c for c in df.columns if c.startswith('metrics.')]}"
            )

        # Drop rows with missing metric values
        df = df.dropna(subset=[metric_col])

        if len(df) < min_sample_size:
            return {
                "status": "insufficient_data",
                "recommendation": f"Insufficient valid data: {len(df)} runs with valid metrics < {min_sample_size} minimum.",
                "sample_size": len(df),
                "metric": metric,
            }

        metric_values = df[metric_col].values

        # Compute summary statistics
        summary_stats = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
            "min": float(np.min(metric_values)),
            "max": float(np.max(metric_values)),
            "range": float(np.max(metric_values) - np.min(metric_values)),
        }

        # Identify best run (for loss metrics, lower is better)
        best_idx = df[metric_col].idxmin()
        best_run_id = df.loc[best_idx, "run_id"]
        best_metric_value = df.loc[best_idx, metric_col]

        # Perform statistical tests
        statistical_tests = []

        if group_by is not None:
            # Group-based comparison
            if group_by not in df.columns:
                raise ValueError(
                    f"Group column {group_by} not found. "
                    f"Available columns: {df.columns.tolist()}"
                )

            groups = df.groupby(group_by)
            group_names = list(groups.groups.keys())

            # Perform pairwise comparisons between groups
            for i, group1_name in enumerate(group_names):
                for group2_name in group_names[i + 1:]:
                    group1_data = groups.get_group(group1_name)[metric_col].values
                    group2_data = groups.get_group(group2_name)[metric_col].values

                    if len(group1_data) >= 2 and len(group2_data) >= 2:
                        test_result = self.perform_statistical_test(
                            group1_data, group2_data, alpha=alpha
                        )
                        test_result["group1"] = group1_name
                        test_result["group2"] = group2_name
                        statistical_tests.append(test_result)
        else:
            # Overall comparison: compare each run to best run
            best_metric_value_array = df.loc[best_idx, metric_col]
            for idx, row in df.iterrows():
                if idx != best_idx:
                    other_metric = row[metric_col]
                    test_result = self.perform_statistical_test(
                        np.array([best_metric_value_array, other_metric]),
                        np.array([other_metric, best_metric_value_array]),
                        alpha=alpha
                    )
                    test_result["run_id"] = row["run_id"]
                    statistical_tests.append(test_result)

        # Determine overall significance
        overall_significant = any(test["significant"] for test in statistical_tests)

        # Generate recommendation
        if overall_significant:
            # Find the most significant test
            significant_tests = [t for t in statistical_tests if t["significant"]]
            best_test = max(significant_tests, key=lambda x: abs(x["effect_size"]))
            recommendation = self._generate_recommendation(
                best_test["p_value"],
                best_test["effect_size"],
                best_test["improvement_pct"],
                alpha=alpha
            )
        else:
            recommendation = "No significant differences detected between groups. Consider running more experiments or adjusting hyperparameters."

        return {
            "status": "success",
            "best_run": best_run_id,
            "best_metric": float(best_metric_value),
            "summary_statistics": summary_stats,
            "statistical_tests": statistical_tests,
            "overall_significant": overall_significant,
            "recommendation": recommendation,
            "sample_size": len(df),
            "metric": metric,
        }

    def compare_hyperparameters(
        self,
        run_ids: List[str],
        metric: str = "val.rmse",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze hyperparameter correlations with performance.

        Computes Pearson correlation between hyperparameters and metrics,
        ranking them by correlation strength.

        Args:
            run_ids: List of run IDs to analyze
            metric: Metric name to analyze (default: "val.rmse")
            top_n: Number of top hyperparameters to return (default: 10)

        Returns:
            DataFrame with columns:
                - parameter: str (parameter name)
                - correlation: float (Pearson correlation with metric)
                - abs_correlation: float (absolute value for sorting)
                - direction: str ("positive" or "negative")
                - interpretation: str (human-readable interpretation)
        """
        # Fetch run data
        df = self.comparator.compare_by_ids(run_ids, as_dataframe=True)

        # Extract parameter and metric columns
        param_cols = [c for c in df.columns if c.startswith("params.")]
        metric_col = f"metrics.{metric}"

        if metric_col not in df.columns:
            raise ValueError(
                f"Metric {metric} not found in runs. "
                f"Available metrics: {[c for c in df.columns if c.startswith('metrics.')]}"
            )

        if not param_cols:
            raise ValueError("No parameter columns found in runs.")

        # Compute correlations
        correlations = []

        for param_col in param_cols:
            # Try to convert param to numeric
            param_series = pd.to_numeric(df[param_col], errors="coerce")
            metric_series = df[metric_col]

            # Drop NaN values
            valid_idx = ~(param_series.isna() | metric_series.isna())

            if valid_idx.sum() < 2:
                continue

            # Compute correlation
            corr = param_series[valid_idx].corr(metric_series[valid_idx], method="pearson")

            if not pd.isna(corr):
                param_name = param_col.replace("params.", "")
                correlations.append({
                    "parameter": param_name,
                    "correlation": float(corr),
                    "abs_correlation": float(abs(corr)),
                    "direction": "positive" if corr > 0 else "negative",
                    "interpretation": self._interpret_correlation(corr, param_name, metric),
                })

        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)

        if not corr_df.empty:
            corr_df = corr_df.sort_values("abs_correlation", ascending=False)
            corr_df = corr_df.head(top_n)
            corr_df = corr_df.reset_index(drop=True)

        return corr_df

    def rank_experiments(
        self,
        run_ids: List[str],
        metrics: List[str] = ["val.rmse", "val.mae"],
        weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Rank experiments by multiple metrics.

        Computes weighted composite scores for experiment ranking.

        Args:
            run_ids: List of run IDs to rank
            metrics: List of metric names to consider (default: ["val.rmse", "val.mae"])
            weights: Optional weights for each metric (equal weights if None)

        Returns:
            DataFrame with columns:
                - run_id: str
                - score: float (weighted composite score)
                - rank: int (1 = best)
                - metrics: dict (individual metric values)
                - normalized_metrics: dict (normalized values)
        """
        # Fetch run data
        df = self.comparator.compare_by_ids(run_ids, as_dataframe=True)

        # Set weights
        if weights is None:
            weights = [1.0 / len(metrics)] * len(metrics)

        if len(weights) != len(metrics):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of metrics ({len(metrics)})")

        # Normalize weights
        weights = [w / sum(weights) for w in weights]

        # Extract metric columns
        metric_cols = [f"metrics.{m}" for m in metrics]

        for metric_col in metric_cols:
            if metric_col not in df.columns:
                raise ValueError(
                    f"Metric {metric_col} not found in runs. "
                    f"Available metrics: {[c for c in df.columns if c.startswith('metrics.')]}"
                )

        # Drop rows with missing metrics
        df = df.dropna(subset=metric_cols)

        # Compute normalized metrics (min-max normalization)
        # For loss metrics, lower is better, so we use (1 - normalized)
        normalized_metrics_dict = {}
        for metric_col, weight in zip(metric_cols, weights):
            metric_values = df[metric_col].values
            min_val = np.min(metric_values)
            max_val = np.max(metric_values)

            if max_val == min_val:
                # All values are the same
                normalized = np.zeros_like(metric_values)
            else:
                # Normalize to 0-1 (lower is better for loss metrics)
                normalized = 1 - (metric_values - min_val) / (max_val - min_val)

            normalized_metrics_dict[metric_col] = normalized

        # Compute weighted scores
        scores = np.zeros(len(df))
        for metric_col, weight in zip(metric_cols, weights):
            scores += normalized_metrics_dict[metric_col] * weight

        # Create result DataFrame
        results = []
        for idx, row in df.iterrows():
            metric_values = {m: row[f"metrics.{m}"] for m in metrics}
            normalized_values = {m: float(normalized_metrics_dict[f"metrics.{m}"][idx]) for m in metrics}

            results.append({
                "run_id": row["run_id"],
                "score": float(scores[idx]),
                "metrics": metric_values,
                "normalized_metrics": normalized_values,
            })

        results_df = pd.DataFrame(results)

        # Sort by score (descending)
        results_df = results_df.sort_values("score", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)
        results_df = results_df.reset_index(drop=True)

        return results_df

    def _interpret_correlation(self, corr: float, param_name: str, metric: str) -> str:
        """
        Generate human-readable correlation interpretation.

        Args:
            corr: Pearson correlation coefficient
            param_name: Parameter name
            metric: Metric name

        Returns:
            Human-readable interpretation string
        """
        abs_corr = abs(corr)

        if abs_corr < 0.3:
            strength = "Weak"
        elif abs_corr < 0.7:
            strength = "Moderate"
        else:
            strength = "Strong"

        # For loss metrics (where lower is better), negative correlation means
        # higher parameter value -> lower metric (better performance)
        if corr < 0:
            direction = "better"
        else:
            direction = "worse"

        return f"{strength} correlation between {param_name} and {metric}. Higher {param_name} associated with {direction} {metric}."
