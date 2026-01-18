"""
MLflow Analytics - Error analysis and failure mode identification.

This module provides tools for analyzing prediction errors, identifying
systematic failure modes, and generating diagnostic visualizations.
"""

from mlflow_tracking.analytics.error_analyzer import ErrorAnalyzer
from mlflow_tracking.analytics.visualizations import (
    plot_residuals,
    plot_error_distribution,
    plot_prediction_vs_actual,
    plot_failure_modes,
)

__all__ = [
    "ErrorAnalyzer",
    "plot_residuals",
    "plot_error_distribution",
    "plot_prediction_vs_actual",
    "plot_failure_modes",
]
