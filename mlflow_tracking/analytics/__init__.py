"""
MLflow Analytics - Error analysis, model interpretability, and insights generation.

This module provides tools for analyzing prediction errors, identifying
systematic failure modes, explaining model predictions using SHAP and ELI5,
and generating automated insights from experiment comparisons.
"""

from mlflow_tracking.analytics.error_analyzer import ErrorAnalyzer
from mlflow_tracking.analytics.visualizations import (
    plot_residuals,
    plot_error_distribution,
    plot_prediction_vs_actual,
    plot_failure_modes,
)
from mlflow_tracking.analytics.interpretability import (
    ModelInterpretability,
    compute_shap,
    plot_feature_importance,
    plot_local_explanation,
    compute_permutation_importance,
)
from mlflow_tracking.analytics.insights_generator import (
    InsightsGenerator,
    generate_insights,
    compare_hyperparameters,
    rank_experiments,
)

__all__ = [
    "ErrorAnalyzer",
    "plot_residuals",
    "plot_error_distribution",
    "plot_prediction_vs_actual",
    "plot_failure_modes",
    "ModelInterpretability",
    "compute_shap",
    "plot_feature_importance",
    "plot_local_explanation",
    "compute_permutation_importance",
    "InsightsGenerator",
    "generate_insights",
    "compare_hyperparameters",
    "rank_experiments",
]
