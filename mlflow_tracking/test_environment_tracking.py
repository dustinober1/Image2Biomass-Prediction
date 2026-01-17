"""
Test environment tracking functionality.

This script demonstrates automatic and manual environment logging
with the ExperimentTracker class.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlflow_tracking import ExperimentTracker


def test_auto_logging():
    """Test auto-logging of environment (default behavior)."""
    print("Testing auto-logging environment...")
    with ExperimentTracker("env_test_auto") as tracker:
        run_id = tracker.start_run("auto_env_run", random_seed=42)

        # Log some experiment params and metrics
        tracker.log_params({"learning_rate": 0.01, "epochs": 100})
        tracker.log_metrics({"train_rmse": 10.5, "val_rmse": 12.3})

        print(f"  Run ID: {run_id}")
        print("  Environment auto-logged (check MLflow UI)")


def test_manual_logging():
    """Test manual environment logging."""
    print("\nTesting manual environment logging...")
    tracker2 = ExperimentTracker("env_test_manual", auto_log_environment=False)
    tracker2.start_run("manual_env_run")

    # Manually log environment
    from mlflow_tracking.environment import log_environment_to_mlflow
    log_environment_to_mlflow()

    tracker2.log_params({"model": "RandomForest", "n_estimators": 50})
    tracker2.log_metrics({"accuracy": 0.95})
    tracker2.end_run()


def test_random_seed_logging():
    """Test random seed is logged correctly."""
    print("\nTesting random seed logging...")
    with ExperimentTracker("env_test_seed") as tracker:
        run_id = tracker.start_run("seeded_run", random_seed=123)
        tracker.log_params({"batch_size": 32})
        tracker.log_metrics({"loss": 0.123})
        print(f"  Run ID: {run_id}")
        print("  Random seed 123 should be logged as param")


if __name__ == "__main__":
    test_auto_logging()
    test_manual_logging()
    test_random_seed_logging()

    print("\n✓ Environment tracking test complete")
    print("Run 'mlflow ui' to verify environment tags and params were logged")
    print("\nExpected tags: git.commit_hash, git.branch, system.os, python.version")
    print("Expected params: env.numpy, env.pandas, random_seed")
