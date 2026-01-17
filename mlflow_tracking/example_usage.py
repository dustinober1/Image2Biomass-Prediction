#!/usr/bin/env python3
"""
Example usage of the ExperimentTracker SDK.

This script demonstrates the full workflow for logging experiments
with MLflow, including both manual and context manager usage.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tempfile

from mlflow_tracking import ExperimentTracker


def demo_manual_tracking():
    """Demonstrate manual experiment tracking workflow."""
    print("\n=== Demo 1: Manual Tracking Workflow ===\n")

    # Create tracker instance
    tracker = ExperimentTracker("test_experiment")

    # Start a new run
    run_id = tracker.start_run("test_run", tags={"purpose": "demo", "model": "xgboost"})
    print(f"Started run: {run_id}")

    # Log hyperparameters
    tracker.log_params({
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 100,
        "subsample": 0.8,
    })
    print("Logged hyperparameters")

    # Log metrics (simulating training results)
    tracker.log_metrics({
        "train_rmse": 10.5,
        "val_rmse": 12.3,
        "r2": 0.85,
        "train_mae": 8.2,
        "val_mae": 9.7,
    })
    print("Logged metrics")

    # Create and log a dummy artifact (model file as text)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Dummy model file\nWeights: [1, 2, 3]\nBias: 0.5\n")
        artifact_path = f.name

    tracker.log_artifact(artifact_path, artifact_path="models")
    print(f"Logged artifact: {artifact_path}")

    # Clean up dummy file
    Path(artifact_path).unlink()

    # End the run
    tracker.end_run(status="completed")
    print("Run completed")


def demo_context_manager():
    """Demonstrate context manager workflow with automatic status tracking."""
    print("\n=== Demo 2: Context Manager Workflow ===\n")

    # Context manager automatically handles run completion/failure
    with ExperimentTracker("context_test") as tracker:
        run_id = tracker.start_run("context_run", tags={"demo": "context_manager"})
        print(f"Started run: {run_id}")

        tracker.log_params({
            "param1": "value1",
            "param2": "value2",
            "param3": 42,
        })
        print("Logged parameters")

        # Simulate some work
        tracker.log_metrics({
            "metric1": 0.95,
            "metric2": 0.87,
            "metric3": 1.23,
        })
        print("Logged metrics")

        # Context manager will automatically call end_run(status="completed")
    print("Context manager exited - run automatically completed")


def demo_error_handling():
    """Demonstrate automatic failure tracking when exceptions occur."""
    print("\n=== Demo 3: Error Handling ===\n")

    try:
        with ExperimentTracker("error_test") as tracker:
            run_id = tracker.start_run("error_run", tags={"demo": "error_handling"})
            print(f"Started run: {run_id}")

            tracker.log_params({"param": "value"})
            tracker.log_metrics({"metric": 0.5})
            print("Logged data before error")

            # Simulate an error
            raise RuntimeError("Simulated training error")
    except RuntimeError:
        print("Error caught - context manager marked run as failed")
    print("Error handling demonstration complete")


def main():
    """Run all demonstration examples."""
    print("=" * 60)
    print("MLflow ExperimentTracker - Example Usage")
    print("=" * 60)

    try:
        # Demo 1: Manual workflow
        demo_manual_tracking()

        # Demo 2: Context manager
        demo_context_manager()

        # Demo 3: Error handling
        demo_error_handling()

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        print("\nTo view logged experiments, run:")
        print("  mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlruns.db")
        print("\nThen open http://localhost:5000 in your browser")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
