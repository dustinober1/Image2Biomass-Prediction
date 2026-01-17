"""
Auto-logging for ML frameworks using MLflow's built-in autolog capabilities.

This module provides the AutoLogger class which enables automatic metric logging
for sklearn, XGBoost, and PyTorch models without requiring manual logging code
in training scripts.

MLflow's autolog automatically captures:
- Parameters: Model hyperparameters, training configuration
- Metrics: Loss curves, accuracy, RMSE, and other training metrics
- Models: Trained model artifacts via mlflow.<framework>.log_model()
- Artifacts: Training outputs, plots, and other files

Framework-specific coverage:
- sklearn: Parameters, metrics, and models for estimators (RFC, Ridge, etc.)
- xgboost: Training/validation metrics, parameters, and model artifacts
- pytorch: Epoch-level metrics, parameters, and PyTorch modules

This satisfies INTEGRATION-02: Automatic metric logging without script modifications.
"""

import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from pathlib import Path
from typing import Optional


class AutoLogger:
    """
    Auto-logging for ML frameworks using MLflow's built-in autolog.

    This class provides a context manager interface to enable framework-specific
    automatic logging during model training. MLflow autolog captures metrics,
    parameters, and models without requiring manual logging code.

    Example:
        >>> # Enable sklearn autolog for training
        >>> with AutoLogger('sklearn'):
        ...     model = RandomForestRegressor()
        ...     model.fit(X_train, y_train)
        ...     # Metrics automatically logged to active MLflow run

        >>> # Enable pytorch autolog with framework detection
        >>> framework = AutoLogger.detect_framework('scripts/train_model.py')
        >>> with AutoLogger(framework):
        ...     train_model(...)  # Metrics logged automatically

    Attributes:
        framework: ML framework name ('sklearn', 'xgboost', 'pytorch')
        _original_loggers: Internal state for logger restoration (reserved)

    Note:
        MLflow autolog requires an active MLflow run. The ExperimentTracker
        context manager should be used before AutoLogger.
    """

    def __init__(self, framework: str):
        """
        Initialize AutoLogger for a specific framework.

        Args:
            framework: ML framework name ('sklearn', 'xgboost', 'pytorch')

        Raises:
            ValueError: If framework is not supported
        """
        self.framework = framework.lower()
        self._original_loggers = None

        # Validate framework support
        supported = ['sklearn', 'xgboost', 'pytorch']
        if self.framework not in supported:
            raise ValueError(
                f"Unknown framework: {self.framework}. "
                f"Supported frameworks: {supported}"
            )

    def __enter__(self):
        """
        Enable autolog for the framework.

        This method is called when entering the context manager and enables
        MLflow's autolog for the specified framework.

        Returns:
            self (allows chaining context managers)

        Example:
            >>> with AutoLogger('sklearn'):
            ...     # sklearn autolog is now active
            ...     model.fit(X, y)
        """
        if self.framework == 'sklearn':
            mlflow.sklearn.autolog()
        elif self.framework == 'xgboost':
            mlflow.xgboost.autolog()
        elif self.framework == 'pytorch':
            mlflow.pytorch.autolog()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Disable autolog after training.

        MLflow autolog is session-based and automatically disables when
        the run ends. No explicit disable is needed here.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        # MLflow autolog is session-based, no explicit disable needed
        # The autolog session ends when the MLflow run ends
        pass

    @staticmethod
    def detect_framework(script_path: str) -> str:
        """
        Detect ML framework from script imports.

        This static method analyzes a Python script's import statements
        to determine which ML framework it uses. This enables automatic
        framework selection when using adapters.

        Args:
            script_path: Path to training script

        Returns:
            Detected framework name ('pytorch', 'xgboost', 'sklearn', or 'unknown')

        Example:
            >>> AutoLogger.detect_framework('scripts/train_tabular_baseline.py')
            'xgboost'
            >>> AutoLogger.detect_framework('scripts/train_oof_effnet.py')
            'pytorch'

        Note:
            Detection is based on import statements:
            - 'import torch' or 'from torch' → 'pytorch'
            - 'import xgboost' or 'from xgboost' → 'xgboost'
            - 'import sklearn' or 'from sklearn' → 'sklearn'

            If multiple frameworks are imported, the first match in the
            priority order (pytorch > xgboost > sklearn) is returned.
        """
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        with open(script_path, 'r') as f:
            content = f.read()

        # Priority order: check torch first, then xgboost, then sklearn
        # This handles cases where scripts might use multiple frameworks
        if 'import torch' in content or 'from torch' in content:
            return 'pytorch'
        elif 'import xgboost' in content or 'from xgboost' in content:
            return 'xgboost'
        elif 'import sklearn' in content or 'from sklearn' in content:
            return 'sklearn'
        else:
            return 'unknown'

    def is_supported(self, script_path: Optional[str] = None) -> bool:
        """
        Check if a framework or script is supported for auto-logging.

        Args:
            script_path: Optional path to script for framework detection.
                        If not provided, checks if self.framework is supported.

        Returns:
            True if framework is supported for auto-logging

        Example:
            >>> logger = AutoLogger('sklearn')
            >>> logger.is_supported()
            True

            >>> AutoLogger('sklearn').is_supported('scripts/train_ridge.py')
            True
        """
        if script_path:
            framework = self.detect_framework(script_path)
            return framework in ['sklearn', 'xgboost', 'pytorch']
        return self.framework in ['sklearn', 'xgboost', 'pytorch']
