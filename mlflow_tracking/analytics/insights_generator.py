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
