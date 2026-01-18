"""
MLflow Analytics - Model interpretability and explanation.

This module provides tools for explaining model predictions using SHAP values
and permutation importance, enabling users to understand feature importance
and individual prediction explanations.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# MLflow imports
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# SHAP imports (lazy loading for graceful error handling)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ELI5 imports (lazy loading for graceful error handling)
try:
    from eli5.sklearn import PermutationImportance
    import eli5
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

from mlflow.exceptions import MlflowException


class ModelInterpretability:
    """
    Model interpretability using SHAP values and permutation importance.

    This class provides methods for:
    - Loading models from MLflow artifacts
    - Computing SHAP values for feature importance
    - Creating feature importance plots (global and local)
    - Computing permutation importance using ELI5
    - Logging explanations as MLflow artifacts

    Example usage:
        >>> interpreter = ModelInterpretability()
        >>> shap_values, explainer = interpreter.compute_shap(run_id="abc123", X_test=X_test)
        >>> fig = interpreter.plot_feature_importance(shap_values, X_test)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize ModelInterpretability with MLflow client.

        Args:
            tracking_uri: Optional MLflow tracking URI. If None, uses default.
        """
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri) if tracking_uri else MlflowClient()

        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for model interpretability. "
                "Install with: pip install shap"
            )

    def _load_model_from_artifacts(self, run_id: str) -> Any:
        """
        Load model from MLflow artifacts.

        Args:
            run_id: MLflow run ID containing the model artifact

        Returns:
            Loaded model object

        Raises:
            MlflowException: If model artifact doesn't exist
        """
        try:
            # Get run info to find model artifacts
            run = self.client.get_run(run_id)
            artifacts = self.client.list_artifacts(run_id)

            # Find model artifacts (MLflow models are logged with MLmodel file)
            model_artifacts = [
                a for a in artifacts
                if a.path.endswith('.yaml') or
                   a.path.endswith('.pkl') or
                   a.path.endswith('.pt') or
                   any(child.path.endswith('MLmodel') for child in
                       [self.client.list_artifacts(run_id, a.path)] if a.is_dir)
            ]

            # Try common model paths
            model_paths_to_try = []

            # Add 'model' as default path (most common)
            model_paths_to_try.append("model")

            # Add any directories that look like model artifacts
            for artifact in artifacts:
                if artifact.is_dir and not artifact.path.startswith('.'):
                    model_paths_to_try.append(artifact.path)

            # Try each path until one works
            last_error = None
            for model_path in model_paths_to_try:
                try:
                    model_uri = f"runs:/{run_id}/{model_path}"
                    model = mlflow.pyfunc.load_model(model_uri)

                    # Extract underlying model for sklearn/xgboost/pytorch
                    # MLflow pyfunc wraps the model, we need the underlying model
                    if hasattr(model, '_model_impl'):
                        # For sklearn and xgboost models
                        underlying_model = model._model_impl
                        if hasattr(underlying_model, 'model'):
                            return underlying_model.model
                        return underlying_model
                    elif hasattr(model, 'model'):
                        # For pytorch models
                        return model.model

                    return model

                except Exception as e:
                    last_error = e
                    continue

            # If all paths failed, raise the last error
            if last_error:
                raise last_error

            raise MlflowException(f"No model artifact found in run {run_id}")

        except Exception as e:
            raise MlflowException(
                f"Failed to load model from run {run_id}: {str(e)}"
            )

    def _detect_model_type(self, model: Any) -> str:
        """
        Detect model type for appropriate explainer selection.

        Args:
            model: Model object

        Returns:
            Model type string: 'tree', 'linear', 'deep', or 'kernel'
        """
        # Check for tree-based models
        if hasattr(model, 'feature_importances_'):
            return 'tree'

        # Check for linear models
        if hasattr(model, 'coef_'):
            return 'linear'

        # Check for PyTorch models
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return 'deep'
        except ImportError:
            pass

        # Default fallback
        return 'kernel'

    def _create_explainer(
        self,
        model: Any,
        X_background: pd.DataFrame,
        model_type: Optional[str] = None
    ) -> 'shap.Explainer':
        """
        Create appropriate SHAP explainer based on model type.

        Args:
            model: Model object
            X_background: Background dataset for explainer
            model_type: Optional model type hint ('tree', 'linear', 'deep', 'kernel')

        Returns:
            SHAP explainer object

        Raises:
            ValueError: If model_type is invalid
        """
        if model_type is None:
            model_type = self._detect_model_type(model)

        # Log explainer type for reproducibility
        import mlflow
        if mlflow.active_run():
            mlflow.log_param("interpretability.explainer_type", model_type)

        if model_type == "tree":
            # TreeExplainer: Optimal for XGBoost, RandomForest, GradientBoosting
            explainer = shap.TreeExplainer(model, data=X_background)
        elif model_type == "linear":
            # LinearExplainer: Optimal for Ridge, Lasso, LinearRegression
            explainer = shap.LinearExplainer(model, X_background)
        elif model_type == "deep":
            # DeepExplainer: For PyTorch models
            explainer = shap.DeepExplainer(model, X_background)
        else:
            # KernelExplainer: Model-agnostic fallback (slower)
            explainer = shap.KernelExplainer(model.predict, X_background)

        return explainer

    def compute_shap(
        self,
        run_id: str,
        X_test: pd.DataFrame,
        background_samples: int = 100,
        model_type: Optional[str] = None
    ) -> Tuple[np.ndarray, 'shap.Explainer']:
        """
        Compute SHAP values for model predictions.

        Args:
            run_id: MLflow run ID containing the model
            X_test: Test dataset features
            background_samples: Number of background samples for explainer
            model_type: Optional model type hint ('tree', 'linear', 'deep', 'kernel')

        Returns:
            Tuple of (shap_values, explainer)

        Raises:
            ValueError: If X_test is empty or background_samples > len(X_test)
            MlflowException: If model artifact doesn't exist
        """
        if len(X_test) == 0:
            raise ValueError("X_test cannot be empty")

        if background_samples > len(X_test):
            raise ValueError(
                f"background_samples ({background_samples}) > len(X_test) ({len(X_test)})"
            )

        # Load model from MLflow
        model = self._load_model_from_artifacts(run_id)

        # Select background samples (random subset)
        X_background = X_test.sample(n=min(background_samples, len(X_test)), random_state=42)

        # Create explainer
        explainer = self._create_explainer(model, X_background, model_type)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)

        return shap_values, explainer

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        plot_type: str = "summary",
        max_features: int = 20
    ) -> plt.Figure:
        """
        Create feature importance visualization from SHAP values.

        Args:
            shap_values: SHAP values from compute_shap()
            X_test: Test dataset features
            feature_names: Optional list of feature names
            plot_type: Type of plot ('summary' or 'bar')
            max_features: Maximum number of features to display

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If plot_type is not 'summary' or 'bar'
        """
        if feature_names is None:
            feature_names = X_test.columns.tolist()

        # Limit to max_features
        if len(feature_names) > max_features:
            # Select features with highest mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-max_features:]
            shap_values_subset = shap_values[:, top_indices]
            feature_names_subset = [feature_names[i] for i in top_indices]
            X_test_subset = X_test.iloc[:, top_indices]
        else:
            shap_values_subset = shap_values
            feature_names_subset = feature_names
            X_test_subset = X_test

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "summary":
            # Summary plot: Shows global feature importance with impact direction
            shap.summary_plot(
                shap_values_subset,
                X_test_subset,
                feature_names=feature_names_subset,
                show=False
            )
            plt.title('Global Feature Importance (SHAP)')
        elif plot_type == "bar":
            # Bar plot: Shows mean absolute SHAP values per feature
            shap.summary_plot(
                shap_values_subset,
                X_test_subset,
                plot_type="bar",
                show=False
            )
            plt.title(f'Mean Absolute SHAP Values (Top {max_features} Features)')
        else:
            raise ValueError(f"plot_type must be 'summary' or 'bar', got '{plot_type}'")

        # Tight layout
        plt.tight_layout()

        return fig

    def plot_local_explanation(
        self,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        sample_idx: int,
        feature_names: Optional[List[str]] = None,
        explainer: Optional['shap.Explainer'] = None
    ) -> plt.Figure:
        """
        Create local explanation (waterfall plot) for single prediction.

        Args:
            shap_values: SHAP values from compute_shap()
            X_test: Test dataset features
            sample_idx: Index of sample to explain
            feature_names: Optional list of feature names
            explainer: Optional SHAP explainer for expected_value

        Returns:
            matplotlib Figure object

        Raises:
            IndexError: If sample_idx is out of range
        """
        if sample_idx < 0 or sample_idx >= len(X_test):
            raise IndexError(
                f"sample_idx {sample_idx} out of range [0, {len(X_test)})"
            )

        if feature_names is None:
            feature_names = X_test.columns.tolist()

        # Extract SHAP values and feature values for sample
        sample_shap_values = shap_values[sample_idx]
        sample_features = X_test.iloc[sample_idx]

        # Get expected value (base value) from explainer or use mean
        if explainer is not None:
            expected_value = explainer.expected_value
        else:
            # Fallback: use mean prediction as base value
            expected_value = np.mean(sample_shap_values)

        # Create explanation object
        explanation = shap.Explanation(
            values=sample_shap_values,
            base_values=expected_value,
            data=sample_features.values,
            feature_names=feature_names
        )

        # Create waterfall plot
        fig = plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'Local Explanation for Sample {sample_idx}')
        plt.tight_layout()

        return fig

    def plot_dependence(
        self,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        feature_idx: int,
        feature_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Create SHAP dependence plot for single feature.

        Shows how feature value affects SHAP value (feature importance),
        colored by interaction feature (auto-selected by SHAP).

        Args:
            shap_values: SHAP values from compute_shap()
            X_test: Test dataset features
            feature_idx: Index of feature to plot
            feature_names: Optional list of feature names

        Returns:
            matplotlib Figure object

        Raises:
            IndexError: If feature_idx is out of range
        """
        if feature_names is None:
            feature_names = X_test.columns.tolist()

        if feature_idx < 0 or feature_idx >= len(feature_names):
            raise IndexError(
                f"feature_idx {feature_idx} out of range [0, {len(feature_names)})"
            )

        # Create dependence plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot: {feature_names[feature_idx]}')
        plt.tight_layout()

        return fig

    def compute_permutation_importance(
        self,
        run_id: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute permutation importance using ELI5.

        Args:
            run_id: MLflow run ID containing the model
            X_test: Test dataset features
            y_test: Test dataset target
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with columns: feature, importance, std (sorted by importance)

        Raises:
            ImportError: If ELI5 is not installed
            ValueError: If model doesn't support sklearn API
            MlflowException: If model artifact doesn't exist
        """
        if not ELI5_AVAILABLE:
            raise ImportError(
                "ELI5 is required for permutation importance. "
                "Install with: pip install eli5"
            )

        # Load model from MLflow
        model = self._load_model_from_artifacts(run_id)

        # Check model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError(
                "Model must have predict() method for permutation importance"
            )

        # Create PermutationImportance instance
        perm = PermutationImportance(
            model,
            scoring='neg_mean_squared_error',
            n_iter=n_repeats,
            random_state=random_state
        )

        # Fit on test data
        perm.fit(X_test, y_test)

        # Extract importance scores directly from the object
        # ELI5 stores results in feature_importances_ and feature_importances_std_
        importances = perm.feature_importances_
        stds = perm.feature_importances_std_

        # Create DataFrame manually
        results_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': importances,
            'std': stds
        })

        # Sort by importance (descending)
        results_df = results_df.sort_values('importance', ascending=False)

        return results_df

    def plot_permutation_importance(
        self,
        perm_importance_df: pd.DataFrame,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create bar plot of permutation importance.

        Args:
            perm_importance_df: DataFrame from compute_permutation_importance()
            top_n: Number of top features to display
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        # Select top_n features
        plot_df = perm_importance_df.head(top_n).copy()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar plot with error bars
        y_positions = range(len(plot_df))
        ax.barh(
            y_positions,
            plot_df['importance'],
            xerr=plot_df['std'],
            alpha=0.8,
            color='steelblue'
        )

        # Set y-axis labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df['feature'])

        # Invert y-axis to show top feature at top
        ax.invert_yaxis()

        # Set labels and title
        ax.set_xlabel('Mean Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title(f'Permutation Importance (Top {top_n} Features)')

        # Add value annotations on bars
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            importance = row['importance']
            ax.text(
                importance,
                i,
                f' {importance:.4f}',
                va='center',
                fontsize=9
            )

        plt.tight_layout()

        return fig


# Convenience functions for common operations
def compute_shap(
    run_id: str,
    X_test: pd.DataFrame,
    background_samples: int = 100,
    model_type: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> Tuple[np.ndarray, 'shap.Explainer']:
    """
    Convenience function to compute SHAP values.

    Args:
        run_id: MLflow run ID
        X_test: Test dataset
        background_samples: Number of background samples
        model_type: Optional model type hint
        tracking_uri: Optional MLflow tracking URI

    Returns:
        Tuple of (shap_values, explainer)
    """
    interpreter = ModelInterpretability(tracking_uri=tracking_uri)
    return interpreter.compute_shap(run_id, X_test, background_samples, model_type)


def plot_feature_importance(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    plot_type: str = "summary",
    max_features: int = 20
) -> plt.Figure:
    """
    Convenience function to plot feature importance.

    Args:
        shap_values: SHAP values
        X_test: Test dataset
        feature_names: Optional feature names
        plot_type: Type of plot ('summary' or 'bar')
        max_features: Maximum features to display

    Returns:
        matplotlib Figure
    """
    interpreter = ModelInterpretability()
    return interpreter.plot_feature_importance(
        shap_values, X_test, feature_names, plot_type, max_features
    )


def plot_local_explanation(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    sample_idx: int,
    feature_names: Optional[List[str]] = None,
    explainer: Optional['shap.Explainer'] = None
) -> plt.Figure:
    """
    Convenience function to plot local explanation.

    Args:
        shap_values: SHAP values
        X_test: Test dataset
        sample_idx: Sample index to explain
        feature_names: Optional feature names
        explainer: Optional SHAP explainer

    Returns:
        matplotlib Figure
    """
    interpreter = ModelInterpretability()
    return interpreter.plot_local_explanation(
        shap_values, X_test, sample_idx, feature_names, explainer
    )


def compute_permutation_importance(
    run_id: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
    tracking_uri: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to compute permutation importance.

    Args:
        run_id: MLflow run ID
        X_test: Test dataset
        y_test: Test target
        n_repeats: Number of permutation repeats
        random_state: Random seed
        tracking_uri: Optional MLflow tracking URI

    Returns:
        DataFrame with feature importance
    """
    interpreter = ModelInterpretability(tracking_uri=tracking_uri)
    return interpreter.compute_permutation_importance(
        run_id, X_test, y_test, n_repeats, random_state
    )
