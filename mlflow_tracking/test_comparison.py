"""
Test script demonstrating experiment comparison and analysis features.

This script creates sample experiments and demonstrates all comparison methods,
export functionality, and insights generation.
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

from mlflow_tracking import ExperimentTracker, ExperimentComparator
from mlflow_tracking.config import MLFLOW_TRACKING_URI


def create_sample_experiments():
    """Create sample experiments with different hyperparameters and metrics."""
    print("=" * 60)
    print("Creating Sample Experiments")
    print("=" * 60)

    # Create a test experiment group
    tracker = ExperimentTracker("comparison_test")

    # Experiment configurations
    configs = [
        # Random Forest experiments
        {"model_type": "random_forest", "n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        {"model_type": "random_forest", "n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"model_type": "random_forest", "n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
        {"model_type": "random_forest", "n_estimators": 100, "max_depth": 20, "min_samples_split": 2},

        # XGBoost experiments
        {"model_type": "xgboost", "n_estimators": 50, "max_depth": 3, "learning_rate": 0.01},
        {"model_type": "xgboost", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"model_type": "xgboost", "n_estimators": 200, "max_depth": 7, "learning_rate": 0.1},

        # Gradient Boosting experiments
        {"model_type": "gradient_boosting", "n_estimators": 50, "max_depth": 3, "learning_rate": 0.01},
        {"model_type": "gradient_boosting", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
    ]

    run_ids = []

    for i, config in enumerate(configs):
        with ExperimentTracker("comparison_test") as tracker:
            run_name = f"{config['model_type']}_run_{i+1}"

            # Start run
            run_id = tracker.start_run(run_name, random_seed=42)

            # Log hyperparameters
            tracker.log_params(config)

            # Simulate training metrics (add some variation)
            np.random.seed(42 + i)
            train_rmse = 5.0 + np.random.randn() * 2.0
            val_rmse = train_rmse + 2.0 + np.random.randn() * 1.0
            test_rmse = val_rmse + 1.0 + np.random.randn() * 0.5

            train_r2 = 0.90 - (train_rmse / 100.0)
            val_r2 = 0.85 - (val_rmse / 100.0)
            test_r2 = 0.82 - (test_rmse / 100.0)

            # Log metrics
            tracker.log_metrics({
                "train.rmse": max(0, train_rmse),
                "val.rmse": max(0, val_rmse),
                "test.rmse": max(0, test_rmse),
                "train.r2": max(0, min(1, train_r2)),
                "val.r2": max(0, min(1, val_r2)),
                "test.r2": max(0, min(1, test_r2)),
            })

            # Add tags
            tracker.add_tags({
                "model_type": config["model_type"],
                "purpose": "comparison_test",
                "status": "completed"
            })

            run_ids.append(run_id)
            print(f"  Created run {i+1}: {run_name[:40]} (val_rmse={val_rmse:.2f})")

    print(f"\nCreated {len(run_ids)} sample experiments")
    print()

    return run_ids


def demo_compare_by_ids(run_ids):
    """Demonstrate comparison by explicit run IDs."""
    print("=" * 60)
    print("1. Compare by Run IDs")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Compare first 5 runs
    subset_ids = run_ids[:5]
    df = comparator.compare_by_ids(subset_ids)

    print(f"\nComparing {len(subset_ids)} runs by ID:")
    print(f"DataFrame shape: {df.shape}")

    # Display key columns
    display_cols = ["run_id", "status"]
    metric_cols = [c for c in df.columns if c.startswith("metrics.") and "rmse" in c]
    param_cols = [c for c in df.columns if c.startswith("params.n_estimators") or c.startswith("params.model_type")]

    cols_to_show = display_cols + metric_cols[:3] + param_cols[:2]
    available_cols = [c for c in cols_to_show if c in df.columns]

    print("\nSample comparison (first 5 columns):")
    print(df[available_cols].head().to_string())

    # Demonstrate dict output
    print("\n--- Dict output format ---")
    results_dict = comparator.compare_by_ids(subset_ids[:2], as_dataframe=False)
    print(f"First run dict keys: {list(results_dict[0].keys())[:10]}...")

    print()


def demo_compare_by_group():
    """Demonstrate comparison by experiment group."""
    print("=" * 60)
    print("2. Compare by Group")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Compare all runs in the test group
    df = comparator.compare_by_group("comparison_test")

    print(f"\nAll runs in 'comparison_test' group: {len(df)} runs")

    # Display summary
    if not df.empty:
        print("\nSummary statistics:")
        metric_cols = [c for c in df.columns if c.startswith("metrics.") and "rmse" in c]
        for col in metric_cols[:3]:
            if col in df.columns:
                print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    print()


def demo_compare_by_filter():
    """Demonstrate comparison by MLflow filter."""
    print("=" * 60)
    print("3. Compare by Filter")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Filter by validation RMSE
    print("\n--- Filter: metrics.val.rmse < 10.0 ---")
    df = comparator.compare_by_filter("metrics.val_rmse < 10.0")
    print(f"Found {len(df)} runs with val.rmse < 10.0")

    # Filter by model type
    print("\n--- Filter: params.model_type = 'random_forest' ---")
    df = comparator.compare_by_filter("params.model_type = 'random_forest'")
    print(f"Found {len(df)} random_forest runs")

    # Combined filter
    print("\n--- Filter: params.model_type = 'xgboost' and metrics.val.rmse < 10.0 ---")
    df = comparator.compare_by_filter(
        "params.model_type = 'xgboost' and metrics.val.rmse < 10.0"
    )
    print(f"Found {len(df)} xgboost runs with val.rmse < 10.0")

    print()


def demo_export_methods(df):
    """Demonstrate export to various formats."""
    print("=" * 60)
    print("4. Export Methods")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Create temporary directory for exports
    temp_dir = tempfile.mkdtemp(prefix="mlflow_comparison_")
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Export to CSV
        csv_path = os.path.join(temp_dir, "comparison.csv")
        comparator.to_csv(df, csv_path)
        print(f"  Exported to CSV: {csv_path}")
        print(f"    File size: {os.path.getsize(csv_path)} bytes")

        # Export to JSON
        json_path = os.path.join(temp_dir, "comparison.json")
        comparator.to_json(df, json_path)
        print(f"  Exported to JSON: {json_path}")
        print(f"    File size: {os.path.getsize(json_path)} bytes")

        # Export to Excel
        try:
            excel_path = os.path.join(temp_dir, "comparison.xlsx")
            comparator.to_excel(df, excel_path)
            print(f"  Exported to Excel: {excel_path}")
            print(f"    File size: {os.path.getsize(excel_path)} bytes")
        except ValueError as e:
            print(f"  Excel export skipped: {e}")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory")

    print()


def demo_validate_metrics(df):
    """Demonstrate metric validation."""
    print("=" * 60)
    print("5. Validate Required Metrics")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Test with all required metrics present
    print("\n--- Validation with all required metrics present ---")
    try:
        comparator.validate_required_metrics(df, ["train.rmse", "val.rmse", "test.rmse"])
        print("  Validation passed: All required metrics present")
    except ValueError as e:
        print(f"  Validation failed: {e}")

    # Test with missing metric
    print("\n--- Validation with missing metric ---")
    try:
        comparator.validate_required_metrics(df, ["train.rmse", "val.rmse", "missing.metric"])
        print("  Validation passed (unexpected)")
    except ValueError as e:
        print(f"  Validation failed (expected): {str(e)[:80]}...")

    print()


def demo_clustering(df):
    """Demonstrate clustering analysis."""
    print("=" * 60)
    print("6. Clustering Analysis")
    print("=" * 60)

    comparator = ExperimentComparator()

    if len(df) < 2:
        print("Not enough runs for clustering (need at least 2)")
        print()
        return

    # Perform clustering
    print("\n--- K-means clustering on metrics ---")
    clusters = comparator.cluster_runs(df, n_clusters=3)

    print(f"Found {clusters['n_clusters']} clusters")
    print(f"  Inertia (within-cluster variance): {clusters['inertia']:.2f}")

    # Show cluster assignments
    print("\n  Cluster assignments:")
    for i, label in enumerate(clusters['cluster_labels']):
        run_id = df.iloc[i]['run_id'][:8]
        print(f"    {run_id} -> Cluster {label}")

    print()


def demo_correlation(df):
    """Demonstrate correlation analysis."""
    print("=" * 60)
    print("7. Correlation Analysis")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Compute correlations
    print("\n--- Parameter-metric correlations ---")
    corr_df = comparator.correlate_params(df, threshold=0.0)

    if not corr_df.empty:
        print(f"Found {len(corr_df)} correlations")

        # Show top correlations
        print("\n  Top correlations (by absolute value):")
        for _, row in corr_df.head(5).iterrows():
            print(f"    {row['param'][:30]:30} <-> {row['metric'][:20]:20} : {row['correlation']:7.3f}")
    else:
        print("No significant correlations found (numeric params required)")

    print()


def demo_outlier_detection(df):
    """Demonstrate outlier detection."""
    print("=" * 60)
    print("8. Outlier Detection")
    print("=" * 60)

    comparator = ExperimentComparator()

    if len(df) < 3:
        print("Not enough runs for outlier detection (need at least 3)")
        print()
        return

    # Z-score method
    print("\n--- Outlier detection using Z-score (threshold=2.0) ---")
    outliers = comparator.find_outliers(df, method="zscore", threshold=2.0)

    print(f"Found {len(outliers['outlier_runs'])} outliers using z-score method")
    if outliers['outlier_runs']:
        print("\n  Outlier runs:")
        for run_id in outliers['outlier_runs']:
            run_data = df[df['run_id'] == run_id]
            score = outliers['outlier_scores'].get(run_id, 0)
            val_rmse = run_data['metrics.val.rmse'].values[0] if 'metrics.val.rmse' in run_data.columns else 'N/A'
            print(f"    {run_id[:8]}: score={score:.2f}, val_rmse={val_rmse}")

    # IQR method
    print("\n--- Outlier detection using IQR (threshold=1.5) ---")
    outliers_iqr = comparator.find_outliers(df, method="iqr", threshold=1.5)

    print(f"Found {len(outliers_iqr['outlier_runs'])} outliers using IQR method")

    print()


def demo_complete_workflow():
    """Demonstrate complete comparison and analysis workflow."""
    print("=" * 60)
    print("9. Complete Workflow Example")
    print("=" * 60)

    comparator = ExperimentComparator()

    # Step 1: Get all runs from group
    print("\nStep 1: Load all experiments from group")
    df = comparator.compare_by_group("comparison_test")
    print(f"  Loaded {len(df)} experiments")

    # Step 2: Filter to best performing runs
    print("\nStep 2: Filter to best performing runs (val.rmse < 10)")
    best_runs = df[df['metrics.val.rmse'] < 10.0] if 'metrics.val.rmse' in df.columns else df
    print(f"  Found {len(best_runs)} runs meeting criteria")

    if len(best_runs) >= 2:
        # Step 3: Cluster similar runs
        print("\nStep 3: Cluster runs by performance")
        clusters = comparator.cluster_runs(best_runs, n_clusters=2)
        print(f"  Identified {clusters['n_clusters']} performance clusters")

        # Step 4: Find key parameter correlations
        print("\nStep 4: Identify key parameter correlations")
        corr = comparator.correlate_params(best_runs, threshold=0.3)
        if not corr.empty:
            print(f"  Top correlation: {corr.iloc[0]['param']} <-> {corr.iloc[0]['metric']}")

        # Step 5: Check for outliers
        print("\nStep 5: Detect anomalous runs")
        outliers = comparator.find_outliers(best_runs, threshold=2.0)
        print(f"  Found {len(outliers['outlier_runs'])} outliers")

    print("\nWorkflow complete!")
    print()


def main():
    """Run all demonstration examples."""
    print("\n" + "=" * 60)
    print("MLflow Experiment Comparison & Analysis Demo")
    print("=" * 60 + "\n")

    # Create sample experiments
    run_ids = create_sample_experiments()

    # Get comparison data for demonstrations
    comparator = ExperimentComparator()
    df = comparator.compare_by_group("comparison_test")

    if df.empty:
        print("No experiments found. Please run the sample creation first.")
        return

    # Run demonstrations
    demo_compare_by_ids(run_ids)
    demo_compare_by_group()
    demo_compare_by_filter()
    demo_export_methods(df)
    demo_validate_metrics(df)
    demo_clustering(df)
    demo_correlation(df)
    demo_outlier_detection(df)
    demo_complete_workflow()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")
    print()


if __name__ == "__main__":
    main()
