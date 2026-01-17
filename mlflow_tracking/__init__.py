"""
MLflow tracking package for experiment logging.

This package provides a Python SDK for systematic experiment tracking
using MLflow with a local SQLite backend.
"""

__version__ = "0.1.0"

# Optional imports - these will be available as modules are added
try:
    from mlflow_tracking.tracker import ExperimentTracker
    _has_tracker = True
except ImportError:
    _has_tracker = False

try:
    from mlflow_tracking.data_split import DataSplitter, create_canonical_splits
    _has_data_split = True
except ImportError:
    _has_data_split = False

# Build __all__ based on what's available
__all__ = []
if _has_tracker:
    __all__.append("ExperimentTracker")
if _has_data_split:
    __all__.extend(["DataSplitter", "create_canonical_splits"])
