"""
Full example: Training a model with complete experiment tracking

This example demonstrates:
1. Loading data and using canonical splits
2. Tracking experiment with ExperimentTracker
3. Logging parameters, metrics, artifacts
4. Automatic environment tracking
5. Error handling and status tracking
6. Preventing cherry-picking (all experiments logged)
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from mlflow_tracking import ExperimentTracker, DataSplitter


def load_data():
    """Load CSIRO biomass dataset

    Returns:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        feature_names: List of feature names
    """
    data_path = Path("csiro-biomass/train.csv")

    if not data_path.exists():
        # Create synthetic data for demonstration if real data not available
        print("Warning: Real data not found, using synthetic data for demonstration")
        n_samples = 357  # Number of images in dataset
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.exponential(scale=50, size=n_samples)  # Positive values like biomass
        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    # Load real data
    df = pd.read_csv(data_path)

    # For this example, we'll predict Dry_Total_g (total dry biomass)
    # Filter for rows where target_name is 'Dry_Total_g'
    biomass_df = df[df['target_name'] == 'Dry_Total_g'].copy()

    # Select numeric features
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    X = biomass_df[feature_cols].values

    # Handle any missing values
    X = np.nan_to_num(X, nan=0.0)

    y = biomass_df['target'].values
    feature_names = feature_cols

    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y, feature_names


def train_model(X_train, y_train, **params):
    """Train RandomForest with given parameters

    Args:
        X_train: Training features
        y_train: Training targets
        **params: Model hyperparameters

    Returns:
        Trained model pipeline
    """
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Evaluate model and return metrics dict

    Args:
        model: Trained model
        X: Features
        y: True targets

    Returns:
        Dictionary with RMSE, MAE, R2 metrics
    """
    y_pred = model.predict(X)
    return {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred)
    }


def run_experiment(experiment_name: str, run_name: str, X, y, **params):
    """
    Run complete experiment with tracking

    Args:
        experiment_name: MLflow experiment name
        run_name: Specific run identifier
        X: Full feature array
        y: Full target array
        **params: Model hyperparameters

    Returns:
        run_id: MLflow run ID
        test_metrics: Dictionary of test set metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}/{run_name}")
    print(f"{'='*60}")

    # Use context manager for automatic success/failure tracking
    with ExperimentTracker(experiment_name, auto_log_environment=True) as tracker:
        # Start run with random seed
        run_id = tracker.start_run(run_name, random_seed=42)

        # Get canonical splits
        splitter = DataSplitter()
        train_idx, val_idx, test_idx = splitter.get_split_indices()

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        print(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Log hyperparameters
        tracker.log_params(params)
        tracker.log_params({
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X.shape[1]
        })

        # Train model (in pipeline to prevent data leakage)
        print(f"Training model with params: {params}")
        model = train_model(X_train, y_train, **params)

        # Evaluate on all splits
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        # Log metrics with prefixes to distinguish splits
        tracker.log_metrics({f"train.{k}": v for k, v in train_metrics.items()})
        tracker.log_metrics({f"val.{k}": v for k, v in val_metrics.items()})
        tracker.log_metrics({f"test.{k}": v for k, v in test_metrics.items()})

        # Save model as artifact
        model_path = f"models/{run_name}_model.pkl"
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        tracker.log_artifact(model_path)

        print(f"\nResults:")
        print(f"  Train RMSE: {train_metrics['rmse']:.3f}, R2: {train_metrics['r2']:.3f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.3f}, R2: {val_metrics['r2']:.3f}")
        print(f"  Test RMSE:  {test_metrics['rmse']:.3f}, R2: {test_metrics['r2']:.3f}")
        print(f"  Run ID: {run_id}")

        return run_id, test_metrics


def main():
    """Run all example experiments"""

    # Load data
    X, y, feature_names = load_data()

    # Example 1: Successful baseline experiment
    print("\n" + "="*60)
    print("Example 1: Successful baseline experiment")
    print("="*60)
    run_id, metrics = run_experiment(
        experiment_name="biomass_prediction",
        run_name="rf_baseline",
        X=X, y=y,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5
    )

    # Example 2: Failed experiment (demonstrates error tracking)
    print("\n" + "="*60)
    print("Example 2: Failed experiment (invalid params)")
    print("="*60)
    try:
        run_id, metrics = run_experiment(
            experiment_name="biomass_prediction",
            run_name="rf_failed",
            X=X, y=y,
            n_estimators=0,  # Invalid: will cause error
            max_depth=10
        )
    except Exception as e:
        print(f"\n  Expected error caught: {e}")
        print("  Experiment marked as 'failed' in MLflow")

    # Example 3: Multiple experiments (demonstrates no cherry-picking)
    print("\n" + "="*60)
    print("Example 3: Running multiple experiments (all logged)")
    print("="*60)

    results = []
    for max_depth in [5, 10, 15, 20]:
        run_id, metrics = run_experiment(
            experiment_name="biomass_prediction_tuning",
            run_name=f"rf_depth_{max_depth}",
            X=X, y=y,
            n_estimators=100,
            max_depth=max_depth
        )
        results.append({
            "run_id": run_id,
            "max_depth": max_depth,
            **{f"test.{k}": v for k, v in metrics.items()}
        })

    # Show all results (not just best) - demonstrates no cherry-picking
    print("\n" + "="*60)
    print("All experiments logged (not cherry-picking):")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df[["max_depth", "test.rmse", "test.r2", "test.mae"]].to_string(index=False))

    # Find best result
    best_idx = results_df["test.rmse"].idxmin()
    best_row = results_df.loc[best_idx]
    print(f"\nBest configuration: max_depth={int(best_row['max_depth'])}, "
          f"test.rmse={best_row['test.rmse']:.3f}")
    print("(Note: All experiments were logged, not just the best one)")

    print("\n" + "="*60)
    print("Full example complete!")
    print("="*60)
    print("\nTo explore logged experiments:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("  3. Check experiments: biomass_prediction, biomass_prediction_tuning")
    print("\nLook for:")
    print("  - Tags: status (running/completed/failed), git.commit_hash, system.os")
    print("  - Params: hyperparameters, env.* (package versions), random_seed")
    print("  - Metrics: train.rmse, val.rmse, test.rmse, train.r2, val.r2, test.r2")
    print("  - Artifacts: model.pkl files")
    print("="*60)


if __name__ == "__main__":
    main()
