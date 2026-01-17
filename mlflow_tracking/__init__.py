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

__all__ = [
    "ExperimentTracker",
    "DataSplitter",
    "create_canonical_splits",
    "get_environment",
    "get_git_hash",
    "get_package_versions",
    "ExperimentOrganizer",
    "create_group",
    "ExperimentComparator"
]
