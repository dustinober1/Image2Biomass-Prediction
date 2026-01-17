"""
Test script demonstrating exp-run CLI usage.

Before running this script:
1. Implement at least one concrete adapter (see 04-04-PLAN.md)
2. Ensure example configs are valid for your adapter
"""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Test CLI help message."""
    print("=== Test 1: CLI Help ===")
    result = subprocess.run(
        ["exp-run", "--help"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    assert result.returncode == 0
    print("Help command works\n")


def test_single_experiment():
    """Test running single experiment."""
    print("=== Test 2: Single Experiment ===")

    # This requires a concrete adapter to be implemented
    result = subprocess.run(
        [
            "exp-run",
            "examples/configs/basic_experiment.yaml",
            "--verbose"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"Note: This test requires implementing a concrete adapter")
        print(f"stderr: {result.stderr}")


def test_sweep_execution():
    """Test running parameter sweep."""
    print("=== Test 3: Parameter Sweep ===")

    result = subprocess.run(
        [
            "exp-run",
            "examples/configs/sweep_experiment.yaml",
            "--sweep",
            "--verbose"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"Note: This test requires implementing a concrete adapter")
        print(f"stderr: {result.stderr}")


def test_programmatic_usage():
    """Test using CLI functions programmatically."""
    print("=== Test 4: Programmatic Usage ===")

    from mlflow_tracking.cli import exp_run_command

    # This requires a concrete adapter to be implemented
    result = exp_run_command(
        config_path="examples/configs/basic_experiment.yaml",
        sweep=False,
        verbose=True
    )

    if result != 0:
        print(f"Note: This test requires implementing a concrete adapter")
    else:
        print(f"Programmatic execution returned: {result}")


if __name__ == "__main__":
    # Test 1: Help (should always work)
    test_cli_help()

    # Tests 2-4: Require concrete adapters (skip with message)
    print("\n--- Tests 2-4 require concrete adapters (see 04-04-PLAN.md) ---\n")

    # Uncomment after implementing adapters:
    # test_single_experiment()
    # test_sweep_execution()
    # test_programmatic_usage()

    print("\nCLI help works. Other tests require adapter implementation.")
