"""
MLflow configuration for experiment tracking.

This module provides configuration settings for MLflow tracking,
including the backend store URI and artifact root directory.
"""

from pathlib import Path
from typing import Dict, Any

# Get the project root directory (parent of mlflow_tracking package)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# MLflow tracking configuration
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow_tracking/mlruns.db"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlflow_tracking" / "artifacts")

# Ensure artifact root directory exists
Path(MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)


def get_mlflow_config() -> Dict[str, str]:
    """
    Get MLflow configuration as a dictionary.

    Returns:
        Dict containing tracking_uri and artifact_root
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_root": MLFLOW_ARTIFACT_ROOT,
    }
