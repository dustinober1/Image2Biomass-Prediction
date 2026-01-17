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
from .config_parser import ExperimentConfig, ConfigParser
from .autolog import AutoLogger
from .seed_manager import SeedManager
from .adapters import BaseAdapter, AdapterRegistry, PyTorchAdapter, SklearnAdapter
from .cli import main, exp_run_command

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
    "AutoLogger",
    "SeedManager",
    "BaseAdapter",
    "AdapterRegistry",
    "PyTorchAdapter",
    "SklearnAdapter",
    "main",
    "exp_run_command"
]
