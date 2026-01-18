"""
MLflow tracking package for experiment logging.

This package provides a Python SDK for systematic experiment tracking
using MLflow with a local SQLite backend.
"""

__version__ = "0.1.0"

# Import main classes and functions
from .tracker import ExperimentTracker
from .data_split import DataSplitter, create_canonical_splits
from .environment import get_environment, get_git_hash, get_package_versions
from .organizer import ExperimentOrganizer, create_group
from .comparison import ExperimentComparator
from .config_parser import ExperimentConfig, ConfigParser, OptimizationConfig, SearchParamConfig
from .autolog import AutoLogger
from .seed_manager import SeedManager
from .adapters import BaseAdapter, AdapterRegistry, PyTorchAdapter, SklearnAdapter
from .cli import main, exp_run_command, main_batch, exp_run_batch_command, main_optimize, exp_run_optimize_command
from .resource_manager import ResourceManager
from .batch_executor import BatchExecutor, ExperimentResult, BatchProgress
from .optuna_optimizer import OptunaOptimizer, create_optimization_config, suggest_params_from_trial

# Import analytics module (error analysis, visualization, and reporting)
from .analytics import (
    ErrorAnalyzer,
    plot_residuals,
    plot_error_distribution,
    plot_prediction_vs_actual,
    plot_failure_modes,
    ModelInterpretability,
    compute_shap,
    plot_feature_importance,
    plot_local_explanation,
    compute_permutation_importance,
    InsightsGenerator,
    generate_insights,
    compare_hyperparameters,
    rank_experiments,
    ReportGenerator,
    generate_html_report,
    generate_pdf_report,
)

__all__ = [
    "ExperimentTracker",
    "DataSplitter",
    "create_canonical_splits",
    "get_environment",
    "get_git_hash",
    "get_package_versions",
    "ExperimentOrganizer",
    "create_group",
    "ExperimentComparator",
    "ExperimentConfig",
    "ConfigParser",
    "OptimizationConfig",
    "SearchParamConfig",
    "AutoLogger",
    "SeedManager",
    "BaseAdapter",
    "AdapterRegistry",
    "PyTorchAdapter",
    "SklearnAdapter",
    "main",
    "exp_run_command",
    "main_batch",
    "exp_run_batch_command",
    "main_optimize",
    "exp_run_optimize_command",
    "ResourceManager",
    "BatchExecutor",
    "ExperimentResult",
    "BatchProgress",
    "OptunaOptimizer",
    "create_optimization_config",
    "suggest_params_from_trial",
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
    "ReportGenerator",
    "generate_html_report",
    "generate_pdf_report",
]
