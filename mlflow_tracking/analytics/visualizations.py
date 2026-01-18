"""
Visualization utilities for error analysis.

This module provides reusable plotting functions for error analysis,
including residual plots, error distributions, and failure mode visualizations.
"""

from typing import Tuple
import warnings

import pandas as pd
import numpy as np


def plot_residuals(predictions_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
    """
    Create residual plot showing prediction errors vs predicted values.

    Uses a regression plot with locally weighted smoothing (LOWESS)
    to visualize trends in prediction errors across the prediction range.

    Args:
        predictions_df: DataFrame with columns [predicted, residual]
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object for artifact logging

    Raises:
        AssertionError: If required columns are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'predicted': [100, 150, 200], 'residual': [5, -10, 3]})
        >>> fig = plot_residuals(df)
        >>> fig.savefig('residuals.png')
    """
    # Validate required columns
    assert 'predicted' in predictions_df.columns, "DataFrame must have 'predicted' column"
    assert 'residual' in predictions_df.columns, "DataFrame must have 'residual' column"

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create residual plot with LOWESS smoothing
    sns.regplot(
        data=predictions_df,
        x='predicted',
        y='residual',
        lowess=True,
        line_kws={'color': 'red', 'lw': 2, 'label': 'LOWESS Trend'},
        scatter_kws={'alpha': 0.5},
        ax=ax
    )

    # Add reference line at y=0 (perfect prediction)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Zero Error')

    # Labels and title
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title('Residual Plot: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_error_distribution(predictions_df: pd.DataFrame, figsize: Tuple[int, int] = (14, 5)) -> 'plt.Figure':
    """
    Create error distribution visualization with histogram and box plot.

    Shows the distribution of residuals to identify bias, skew, and outliers.

    Args:
        predictions_df: DataFrame with columns [residual]
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object with 2 subplots

    Raises:
        AssertionError: If required columns are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'residual': [5, -10, 3, -8, 12]})
        >>> fig = plot_error_distribution(df)
        >>> fig.savefig('error_distribution.png')
    """
    # Validate required columns
    assert 'residual' in predictions_df.columns, "DataFrame must have 'residual' column"

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")

    # Drop NaN values for cleaner visualization
    residuals = predictions_df['residual'].dropna()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: Histogram with KDE
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    sns.histplot(data=predictions_df, x='residual', kde=True, ax=axes[0])
    axes[0].axvline(mean_residual, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_residual:.2f}')

    # Add statistics text box
    textstr = f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.95, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

    axes[0].set_xlabel('Residual', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()

    # Subplot 2: Box plot
    sns.boxplot(data=predictions_df, y='residual', ax=axes[1])
    sns.stripplot(data=predictions_df, y='residual', color='black', alpha=0.3,
                  size=3, jitter=True, ax=axes[1])

    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Error Distribution (Box Plot)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(predictions_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
    """
    Create prediction vs actual scatter plot with R² annotation.

    Shows how well predictions match ground truth values. Points along
    the diagonal line indicate perfect predictions.

    Args:
        predictions_df: DataFrame with columns [actual, predicted, abs_residual]
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object for artifact logging

    Raises:
        AssertionError: If required columns are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'actual': [100, 150, 200],
        ...     'predicted': [95, 160, 195],
        ...     'abs_residual': [5, 10, 5]
        ... })
        >>> fig = plot_prediction_vs_actual(df)
        >>> fig.savefig('pred_vs_actual.png')
    """
    # Validate required columns
    assert 'actual' in predictions_df.columns, "DataFrame must have 'actual' column"
    assert 'predicted' in predictions_df.columns, "DataFrame must have 'predicted' column"
    assert 'abs_residual' in predictions_df.columns, "DataFrame must have 'abs_residual' column"

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot with color mapping by absolute error
    scatter = sns.scatterplot(
        data=predictions_df,
        x='actual',
        y='predicted',
        hue='abs_residual',
        palette='coolwarm_r',
        alpha=0.6,
        ax=ax
    )

    # Add diagonal reference line (perfect prediction)
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

    # Compute R²
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values

    # Drop NaN values for R² calculation
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[valid_mask]
    predicted_clean = predicted[valid_mask]

    if len(actual_clean) > 0:
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)
        ss_tot = np.sum((actual_clean - actual_clean.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        r2 = 0.0

    # Add R² annotation
    ax.text(
        0.05, 0.95,
        f'R² = {r2:.4f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
    )

    # Labels and title
    ax.set_xlabel('Actual Biomass (g)', fontsize=12)
    ax.set_ylabel('Predicted Biomass (g)', fontsize=12)
    ax.set_title('Prediction vs Actual: Model Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig


def plot_failure_modes(clustered_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> 'plt.Figure':
    """
    Create 4-panel failure mode visualization.

    Shows multiple views of clustered failure modes to understand
    systematic error patterns.

    Args:
        clustered_df: DataFrame with columns [actual, predicted, residual, abs_residual, cluster]
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object with 4 subplots (2x2 grid)

    Raises:
        AssertionError: If required columns are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'actual': [100, 150, 200],
        ...     'predicted': [80, 170, 180],
        ...     'residual': [20, -20, 20],
        ...     'abs_residual': [20, 20, 20],
        ...     'cluster': [0, 1, 2]
        ... })
        >>> fig = plot_failure_modes(df)
        >>> fig.savefig('failure_modes.png')
    """
    # Validate required columns
    required_cols = ['actual', 'predicted', 'residual', 'abs_residual', 'cluster']
    for col in required_cols:
        assert col in clustered_df.columns, f"DataFrame must have '{col}' column"

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")

    # Drop NaN values for cleaner visualization
    df_clean = clustered_df.dropna(subset=required_cols).copy()

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Use consistent color palette (tab10)
    palette = 'tab10'
    n_clusters = df_clean['cluster'].nunique()

    # Subplot 1: Scatter of actual vs predicted colored by cluster
    sns.scatterplot(
        data=df_clean,
        x='actual',
        y='predicted',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        ax=axes[0, 0]
    )

    # Add diagonal reference line
    min_val = min(df_clean['actual'].min(), df_clean['predicted'].min())
    max_val = max(df_clean['actual'].max(), df_clean['predicted'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5)

    axes[0, 0].set_xlabel('Actual Values', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Values', fontsize=11)
    axes[0, 0].set_title('Actual vs Predicted by Cluster', fontsize=12, fontweight='bold')
    axes[0, 0].legend(title='Cluster', fontsize=9)

    # Subplot 2: Box plot of residuals by cluster
    sns.boxplot(data=df_clean, x='cluster', y='residual', palette=palette, ax=axes[0, 1])
    sns.stripplot(data=df_clean, x='cluster', y='residual', color='black',
                  alpha=0.3, size=3, jitter=True, ax=axes[0, 1])

    axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Cluster', fontsize=11)
    axes[0, 1].set_ylabel('Residual', fontsize=11)
    axes[0, 1].set_title('Residuals by Cluster', fontsize=12, fontweight='bold')

    # Subplot 3: Bar plot of cluster sizes
    cluster_counts = df_clean['cluster'].value_counts().sort_index()
    axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette(palette, n_clusters))
    axes[1, 0].set_xlabel('Cluster', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(cluster_counts.index)

    # Add count annotations on bars
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        axes[1, 0].text(cluster, count, str(count), ha='center', va='bottom', fontsize=10)

    # Subplot 4: Heatmap of cluster characteristics
    cluster_stats = df_clean.groupby('cluster')[['actual', 'predicted', 'abs_residual']].mean()

    # Normalize heatmap values for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_stats_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_stats),
        index=cluster_stats.index,
        columns=cluster_stats.columns
    )

    sns.heatmap(cluster_stats_normalized.T, annot=cluster_stats.T, fmt='.2f',
                cmap='viridis', cbar_kws={'label': 'Normalized Value'}, ax=axes[1, 1])

    axes[1, 1].set_xlabel('Cluster', fontsize=11)
    axes[1, 1].set_ylabel('Metric', fontsize=11)
    axes[1, 1].set_title('Cluster Characteristics', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
