"""
Test suite for ErrorAnalyzer and visualization utilities.

This script creates sample predictions and demonstrates all error analysis
features including residual computation, visualization, and failure mode
identification.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import modules directly to avoid circular import
import importlib.util

# Load ErrorAnalyzer directly
spec = importlib.util.spec_from_file_location(
    'error_analyzer',
    Path(__file__).parent / 'analytics' / 'error_analyzer.py'
)
error_analyzer_module = importlib.util.module_from_spec(spec)

# Mock the config import
class MockConfig:
    MLFLOW_TRACKING_URI = 'sqlite:///tmp/test_mlflow.db'

sys.modules['mlflow_tracking.config'] = MockConfig
spec.loader.exec_module(error_analyzer_module)

ErrorAnalyzer = error_analyzer_module.ErrorAnalyzer

# Load visualizations directly
spec = importlib.util.spec_from_file_location(
    'visualizations',
    Path(__file__).parent / 'analytics' / 'visualizations.py'
)
visualizations_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualizations_module)

plot_residuals = visualizations_module.plot_residuals
plot_error_distribution = visualizations_module.plot_error_distribution
plot_prediction_vs_actual = visualizations_module.plot_prediction_vs_actual
plot_failure_modes = visualizations_module.plot_failure_modes

# Load ExperimentTracker directly
spec = importlib.util.spec_from_file_location(
    'tracker',
    Path(__file__).parent / 'tracker.py'
)
tracker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker_module)

ExperimentTracker = tracker_module.ExperimentTracker


def test_1_initialization():
    """Test ErrorAnalyzer initialization."""
    print("=" * 60)
    print("Test 1: ErrorAnalyzer Initialization")
    print("=" * 60)

    analyzer = ErrorAnalyzer()

    # Verify client is initialized
    assert analyzer.client is not None, "Client should be initialized"
    print("  ✓ MLflow client initialized")

    # Verify predictions_df and residuals start as None
    assert analyzer.predictions_df is None, "predictions_df should start as None"
    assert analyzer.residuals is None, "residuals should start as None"
    print("  ✓ predictions_df and residuals start as None")

    print("  Test 1 PASSED\n")


def test_2_load_run():
    """Test loading predictions from MLflow artifacts."""
    print("=" * 60)
    print("Test 2: Load Run from MLflow Artifacts")
    print("=" * 60)

    # Create synthetic predictions CSV
    predictions_data = pd.DataFrame({
        'image_id': [f'image_{i:04d}' for i in range(100)],
        'actual': np.random.uniform(50, 300, 100),
        'predicted': np.random.uniform(50, 300, 100)
    })

    # Make predictions somewhat correlated with actual
    predictions_data['predicted'] = predictions_data['actual'] * 0.9 + np.random.normal(0, 10, 100)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    predictions_file = os.path.join(temp_dir, 'predictions.csv')
    predictions_data.to_csv(predictions_file, index=False)

    # Create MLflow run and log artifact
    tracker = ExperimentTracker("error_analyzer_test")

    with tracker:
        run_id = tracker.start_run("test_run", random_seed=42)
        tracker.log_artifact(predictions_file, artifact_path="")

    # Load run using ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    analyzer.load_run(run_id, "predictions.csv")

    # Verify predictions_df is loaded correctly
    assert analyzer.predictions_df is not None, "predictions_df should be loaded"
    assert len(analyzer.predictions_df) == 100, f"Expected 100 rows, got {len(analyzer.predictions_df)}"
    print(f"  ✓ Loaded {len(analyzer.predictions_df)} predictions")

    # Verify residuals are computed
    assert 'residual' in analyzer.predictions_df.columns, "residual column should exist"
    assert 'abs_residual' in analyzer.predictions_df.columns, "abs_residual column should exist"
    print("  ✓ Residuals computed (actual - predicted)")

    # Verify residual computation is correct
    expected_residual = predictions_data['actual'].iloc[0] - predictions_data['predicted'].iloc[0]
    actual_residual = analyzer.predictions_df['residual'].iloc[0]
    assert abs(expected_residual - actual_residual) < 1e-6, "Residual computation incorrect"
    print("  ✓ Residual computation is correct")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("  Test 2 PASSED\n")

    return analyzer


def test_3_residual_plotting(analyzer):
    """Test residual plotting."""
    print("=" * 60)
    print("Test 3: Residual Plotting")
    print("=" * 60)

    # Call plot_residuals and verify Figure is returned
    fig = analyzer.plot_residuals()

    assert fig is not None, "plot_residuals should return a Figure"
    print("  ✓ plot_residuals returned Figure")

    # Verify figure has correct title
    axes = fig.get_axes()
    assert len(axes) > 0, "Figure should have at least one axis"
    title = axes[0].get_title()
    assert "Residual" in title, f"Expected title to contain 'Residual', got: {title}"
    print(f"  ✓ Figure has correct title: '{title}'")

    # Save figure to temporary file and verify it exists
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'residuals.png')
    fig.savefig(temp_file)
    assert os.path.exists(temp_file), "Figure file should be saved"
    print("  ✓ Figure saved to file successfully")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("  Test 3 PASSED\n")


def test_4_error_distribution_plotting(analyzer):
    """Test error distribution plotting."""
    print("=" * 60)
    print("Test 4: Error Distribution Plotting")
    print("=" * 60)

    # Call plot_error_distribution and verify Figure is returned
    fig = analyzer.plot_error_distribution()

    assert fig is not None, "plot_error_distribution should return a Figure"
    print("  ✓ plot_error_distribution returned Figure")

    # Verify figure has 2 subplots
    axes = fig.get_axes()
    assert len(axes) == 2, f"Expected 2 subplots, got {len(axes)}"
    print("  ✓ Figure has 2 subplots (histogram and box plot)")

    # Verify annotations (mean, std) are present in one of the subplots
    has_annotations = False
    for ax in axes:
        texts = ax.texts
        if len(texts) > 0:
            has_annotations = True
            break

    assert has_annotations, "Figure should have annotations (mean/std)"
    print("  ✓ Annotations (mean/std) are present")

    print("  Test 4 PASSED\n")


def test_5_prediction_vs_actual_plotting(analyzer):
    """Test prediction vs actual plotting."""
    print("=" * 60)
    print("Test 5: Prediction vs Actual Plotting")
    print("=" * 60)

    # Call plot_prediction_vs_actual and verify Figure is returned
    fig = analyzer.plot_prediction_vs_actual()

    assert fig is not None, "plot_prediction_vs_actual should return a Figure"
    print("  ✓ plot_prediction_vs_actual returned Figure")

    # Verify scatter plot has diagonal reference line
    axes = fig.get_axes()
    assert len(axes) > 0, "Figure should have at least one axis"

    # Check for diagonal reference line (should have 2+ lines: scatter and reference)
    lines = axes[0].get_lines()
    assert len(lines) >= 1, "Should have reference line"
    print("  ✓ Scatter plot has diagonal reference line")

    # Verify R² annotation is present
    texts = axes[0].texts
    assert len(texts) > 0, "Should have R² annotation"
    assert "R²" in str(texts[0]) or "R2" in str(texts[0]), "Text should contain R²"
    print("  ✓ R² annotation is present")

    print("  Test 5 PASSED\n")


def test_6_failure_mode_identification(analyzer):
    """Test failure mode identification using clustering."""
    print("=" * 60)
    print("Test 6: Failure Mode Identification")
    print("=" * 60)

    # Create synthetic data with clear error clusters
    np.random.seed(42)

    # Cluster 0: Under-predictions for high actual values
    cluster_0 = pd.DataFrame({
        'image_id': [f'cluster0_{i}' for i in range(20)],
        'actual': np.random.uniform(200, 300, 20),
        'predicted': np.random.uniform(150, 200, 20),
    })

    # Cluster 1: Over-predictions for low actual values
    cluster_1 = pd.DataFrame({
        'image_id': [f'cluster1_{i}' for i in range(20)],
        'actual': np.random.uniform(50, 100, 20),
        'predicted': np.random.uniform(120, 170, 20),
    })

    # Cluster 2: Random errors
    cluster_2 = pd.DataFrame({
        'image_id': [f'cluster2_{i}' for i in range(20)],
        'actual': np.random.uniform(100, 200, 20),
        'predicted': np.random.uniform(90, 210, 20),
    })

    # Combine clusters
    combined_data = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)

    # Compute residuals
    combined_data['residual'] = combined_data['actual'] - combined_data['predicted']
    combined_data['abs_residual'] = combined_data['residual'].abs()

    # Create new analyzer with clustered data
    analyzer_clustered = ErrorAnalyzer()
    analyzer_clustered.predictions_df = combined_data
    analyzer_clustered.residuals = combined_data['residual'].values

    # Call identify_failure_modes
    failure_modes = analyzer_clustered.identify_failure_modes(n_clusters=3, error_threshold=20)

    # Verify cluster assignments exist
    assert 'cluster' in failure_modes.columns, "Should have cluster assignments"
    print("  ✓ Cluster assignments exist")

    # Verify cluster statistics are computed
    assert 'cluster_mean_abs_residual' in failure_modes.columns, "Should have cluster statistics"
    print("  ✓ Cluster statistics computed")

    # Verify high-error samples are clustered
    n_clusters = failure_modes['cluster'].nunique()
    assert n_clusters == 3, f"Expected 3 clusters, got {n_clusters}"
    print(f"  ✓ High-error samples clustered into {n_clusters} clusters")

    # Verify cluster counts
    cluster_counts = failure_modes['cluster'].value_counts()
    print(f"  ✓ Cluster sizes: {cluster_counts.to_dict()}")

    print("  Test 6 PASSED\n")


def test_7_error_statistics(analyzer):
    """Test error statistics computation."""
    print("=" * 60)
    print("Test 7: Error Statistics Computation")
    print("=" * 60)

    # Call get_error_statistics
    stats = analyzer.get_error_statistics()

    # Verify dict has all required keys
    required_keys = [
        'mean_abs_error', 'median_abs_error', 'max_abs_error', 'std_residual',
        'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'skewness', 'kurtosis'
    ]

    for key in required_keys:
        assert key in stats, f"Missing required key: {key}"
    print(f"  ✓ All {len(required_keys)} required statistics present")

    # Verify percentiles are computed correctly (monotonically increasing)
    assert stats['p25'] <= stats['p50'] <= stats['p75'], "Percentiles should be monotonically increasing"
    print("  ✓ Percentiles computed correctly (monotonically increasing)")

    # Verify skew and kurtosis are included
    assert 'skewness' in stats, "Should include skewness"
    assert 'kurtosis' in stats, "Should include kurtosis"
    print("  ✓ Skewness and kurtosis included")

    # Print some statistics
    print(f"  Mean Absolute Error: {stats['mean_abs_error']:.2f}")
    print(f"  Median Absolute Error: {stats['median_abs_error']:.2f}")
    print(f"  95th Percentile: {stats['p95']:.2f}")

    print("  Test 7 PASSED\n")


def test_8_visualization_functions():
    """Test standalone visualization functions."""
    print("=" * 60)
    print("Test 8: Standalone Visualization Functions")
    print("=" * 60)

    # Create synthetic data
    df = pd.DataFrame({
        'predicted': np.random.uniform(50, 300, 100),
        'residual': np.random.normal(0, 15, 100),
        'actual': np.random.uniform(50, 300, 100),
        'abs_residual': np.random.uniform(0, 50, 100),
        'cluster': np.random.choice([0, 1, 2], 100)
    })

    # Test plot_residuals
    fig1 = plot_residuals(df)
    assert fig1 is not None, "plot_residuals should return Figure"
    print("  ✓ plot_residuals returns Figure")

    # Test plot_error_distribution
    fig2 = plot_error_distribution(df)
    assert fig2 is not None, "plot_error_distribution should return Figure"
    print("  ✓ plot_error_distribution returns Figure")

    # Test plot_prediction_vs_actual
    fig3 = plot_prediction_vs_actual(df)
    assert fig3 is not None, "plot_prediction_vs_actual should return Figure"
    print("  ✓ plot_prediction_vs_actual returns Figure")

    # Test plot_failure_modes
    fig4 = plot_failure_modes(df)
    assert fig4 is not None, "plot_failure_modes should return Figure"
    print("  ✓ plot_failure_modes returns Figure")

    print("  Test 8 PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ERROR ANALYZER TEST SUITE")
    print("=" * 60 + "\n")

    try:
        # Test 1: Initialization
        test_1_initialization()

        # Test 2: Load run (returns analyzer for subsequent tests)
        analyzer = test_2_load_run()

        # Test 3: Residual plotting
        test_3_residual_plotting(analyzer)

        # Test 4: Error distribution plotting
        test_4_error_distribution_plotting(analyzer)

        # Test 5: Prediction vs actual plotting
        test_5_prediction_vs_actual_plotting(analyzer)

        # Test 6: Failure mode identification
        test_6_failure_mode_identification(analyzer)

        # Test 7: Error statistics
        test_7_error_statistics(analyzer)

        # Test 8: Standalone visualization functions
        test_8_visualization_functions()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
