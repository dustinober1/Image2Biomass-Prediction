"""
Test script demonstrating MLflow experiment organization and discovery features.

This script shows how to:
1. Create experiment groups for organizing related experiments
2. Tag experiments with model type, phase, and purpose
3. Search experiments by metrics, parameters, and tags
4. Find best runs within a group
5. List all available groups

Usage:
    python mlflow_tracking/test_organization.py
"""

import numpy as np
from mlflow_tracking import ExperimentTracker, ExperimentOrganizer


def generate_synthetic_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create target with some signal + noise
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

    return X, y


def train_model(model_type, X_train, y_train, X_val, y_val, **params):
    """
    Train a simple model and return metrics.

    This is a simplified mock training function for demonstration.
    In real usage, this would train actual ML models.
    """
    # Simulate training with different characteristics based on model type
    np.random.seed(params.get('random_seed', 42))

    if model_type == 'random_forest':
        # Simulate RF behavior
        n_estimators = params.get('n_estimators', 100)
        base_rmse = 10.0 - (n_estimators / 100) * 2
    elif model_type == 'xgboost':
        # Simulate XGBoost behavior
        learning_rate = params.get('learning_rate', 0.1)
        base_rmse = 8.0 - (learning_rate * 10)
    else:
        base_rmse = 12.0

    # Add some noise
    train_rmse = base_rmse + np.random.randn() * 0.5
    val_rmse = base_rmse + np.random.randn() * 0.5
    test_rmse = base_rmse + np.random.randn() * 0.5

    return {
        'train_rmse': max(0, train_rmse),
        'val_rmse': max(0, val_rmse),
        'test_rmse': max(0, test_rmse),
        'r2': 0.8 + np.random.randn() * 0.1
    }


def main():
    """Demonstrate organization and discovery features."""
    print("=" * 80)
    print("MLflow Organization and Discovery Test")
    print("=" * 80)
    print()

    # Generate synthetic data
    print("1. Generating synthetic data...")
    X, y = generate_synthetic_data()
    n_train = int(0.6 * len(X))
    n_val = int(0.2 * len(X))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print()

    # Create organizer
    organizer = ExperimentOrganizer()

    # ========================================================================
    # Demonstrate grouping
    # ========================================================================
    print("2. Creating experiment groups...")
    print()

    # Create two groups
    ablation_id = organizer.create_group(
        "ablation-studies",
        tags={"purpose": "feature_ablation", "project": "biomass"}
    )
    print(f"   Created group 'ablation-studies' (ID: {ablation_id})")

    ensemble_id = organizer.create_group(
        "ensemble-tests",
        tags={"purpose": "ensemble_methods", "project": "biomass"}
    )
    print(f"   Created group 'ensemble-tests' (ID: {ensemble_id})")
    print()

    # ========================================================================
    # Demonstrate tagging and grouping with experiments
    # ========================================================================
    print("3. Running experiments with grouping and tagging...")
    print()

    experiments = [
        # Ablation studies
        {
            'group': 'ablation-studies',
            'name': 'rf_baseline',
            'model_type': 'random_forest',
            'purpose': 'baseline',
            'params': {'n_estimators': 100, 'max_depth': 10, 'random_seed': 42}
        },
        {
            'group': 'ablation-studies',
            'name': 'rf_no_feature_1',
            'model_type': 'random_forest',
            'purpose': 'feature_ablation',
            'params': {'n_estimators': 100, 'max_depth': 10, 'random_seed': 42}
        },
        {
            'group': 'ablation-studies',
            'name': 'xgb_baseline',
            'model_type': 'xgboost',
            'purpose': 'baseline',
            'params': {'learning_rate': 0.1, 'n_estimators': 100, 'random_seed': 42}
        },
        # Ensemble tests
        {
            'group': 'ensemble-tests',
            'name': 'rf_xgb_ensemble',
            'model_type': 'ensemble',
            'purpose': 'ensemble_methods',
            'params': {'n_estimators': 50, 'learning_rate': 0.05, 'random_seed': 42}
        },
    ]

    run_ids = []
    for exp_config in experiments:
        group = exp_config['group']
        name = exp_config['name']
        model_type = exp_config['model_type']
        purpose = exp_config['purpose']
        params = exp_config['params']

        print(f"   Running: {name} in group '{group}'")

        # Create tracker with group
        tracker = ExperimentTracker(group)
        tracker.set_group(group)

        # Start run
        tracker.start_run(name, random_seed=params['random_seed'])

        # Add tags
        tracker.add_tags({
            'model_type': model_type,
            'purpose': purpose,
            'phase': 'development'
        })

        # Log params
        tracker.log_params(params)

        # Train and log metrics
        metrics = train_model(model_type, X_train, y_train, X_val, y_val, **params)
        tracker.log_metrics({
            'train.rmse': metrics['train_rmse'],
            'val.rmse': metrics['val_rmse'],
            'test.rmse': metrics['test_rmse'],
            'train.r2': metrics['r2']
        })

        # Get run ID for later
        run_id = tracker.get_run_id()
        run_ids.append(run_id)

        tracker.end_run()
        print(f"     Run ID: {run_id}")
        print(f"     Val RMSE: {metrics['val_rmse']:.2f}")

    print()
    print(f"   Completed {len(run_ids)} experiments")
    print()

    # ========================================================================
    # Demonstrate search
    # ========================================================================
    print("4. Demonstrating search capabilities...")
    print()

    # Search all runs
    print("   a) Search all runs (no filter):")
    all_runs = organizer.search_runs(max_results=100)
    print(f"      Found {len(all_runs)} total runs")
    print()

    # Search by tag
    print("   b) Search by model_type tag:")
    rf_runs = organizer.search_runs(
        filter_string="tags.model_type = 'random_forest'"
    )
    print(f"      Found {len(rf_runs)} random_forest runs")
    for run in rf_runs:
        print(f"        - {run['run_id'][:8]}: {run.get('tags', {}).get('purpose', 'N/A')}")
    print()

    # Search by metric
    print("   c) Search by validation RMSE:")
    good_runs = organizer.search_runs(
        filter_string="metrics.val_rmse < 10.0"
    )
    print(f"      Found {len(good_runs)} runs with val_rmse < 10.0")
    for run in good_runs:
        val_rmse = run['metrics'].get('val_rmse', 'N/A')
        print(f"        - {run['run_id'][:8]}: val_rmse={val_rmse}")
    print()

    # Combined search
    print("   d) Combined search (model_type AND metric):")
    xgb_runs = organizer.search_runs(
        filter_string="tags.model_type = 'xgboost' and metrics.test_rmse < 10.0"
    )
    print(f"      Found {len(xgb_runs)} xgboost runs with test_rmse < 10.0")
    for run in xgb_runs:
        test_rmse = run['metrics'].get('test_rmse', 'N/A')
        print(f"        - {run['run_id'][:8]}: test_rmse={test_rmse}")
    print()

    # ========================================================================
    # Demonstrate best runs
    # ========================================================================
    print("5. Finding best runs in group...")
    print()

    print("   Top 3 runs in 'ablation-studies' by val_rmse:")
    best_runs = organizer.get_best_runs('ablation-studies', 'val_rmse', top_k=3)
    for i, run in enumerate(best_runs, 1):
        run_id_short = run['run_id'][:8]
        val_rmse = run['metrics'].get('val_rmse', 'N/A')
        model_type = run.get('tags', {}).get('model_type', 'N/A')
        purpose = run.get('tags', {}).get('purpose', 'N/A')
        print(f"     {i}. Run {run_id_short}")
        print(f"        Model: {model_type}, Purpose: {purpose}")
        print(f"        Val RMSE: {val_rmse}")
    print()

    # ========================================================================
    # List groups
    # ========================================================================
    print("6. Listing all experiment groups...")
    print()

    groups = organizer.list_groups()
    print(f"   Found {len(groups)} groups:")
    for group in groups:
        tags = group.get('tags', {})
        print(f"     - {group['name']}")
        if tags:
            print(f"       Tags: {tags}")
    print()

    # ========================================================================
    # MLflow UI command
    # ========================================================================
    print("7. MLflow UI")
    print()
    print("   To view experiments in the MLflow UI, run:")
    print("   ```bash")
    print("   mlflow ui")
    print("   ```")
    print("   Then open http://localhost:5000 in your browser")
    print()
    print("   In the UI, you can:")
    print("     - View experiments by group (left sidebar)")
    print("     - Compare runs side-by-side")
    print("     - Filter by metrics, params, and tags")
    print("     - Download artifacts")
    print()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
