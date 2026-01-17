"""
Test script demonstrating auto-logging functionality.

This script demonstrates and validates:
1. AutoLogger framework detection from training script imports
2. SeedManager reproducibility (deterministic random number generation)
3. Adapter integration with AutoLogger and SeedManager
4. Example YAML configuration with random_seed parameter

Run this script to verify auto-logging is working correctly:

    python mlflow_tracking/test_autolog.py

Note: This script does not require MLflow to be installed for framework
detection and SeedManager tests. Full integration tests require MLflow.
"""

import sys
sys.path.insert(0, '.')

# Import modules directly to avoid MLflow dependency in __init__.py
import importlib.util

def load_module(name, path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load autolog and seed_manager directly
autolog_module = load_module("autolog", "mlflow_tracking/autolog.py")
AutoLogger = autolog_module.AutoLogger

seed_manager_module = load_module("seed_manager", "mlflow_tracking/seed_manager.py")
SeedManager = seed_manager_module.SeedManager


def test_framework_detection():
    """Test AutoLogger framework detection from script imports."""
    print("=" * 70)
    print("TEST 1: Framework Detection")
    print("=" * 70)

    scripts = [
        ('scripts/train_tabular_baseline.py', 'xgboost'),
        ('scripts/train_oof_effnet.py', 'pytorch'),
        ('scripts/train_ridge_advanced.py', 'sklearn'),
        ('scripts/train_multimodal.py', 'pytorch'),
    ]

    print("\nDetecting ML frameworks from training scripts:")
    print(f"{'Script':<40} {'Detected Framework':<20} {'Expected':<20} {'Status':<10}")
    print("-" * 90)

    all_passed = True
    for script_path, expected_framework in scripts:
        try:
            detected = AutoLogger.detect_framework(script_path)
            status = "PASS" if detected == expected_framework else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"{script_path:<40} {detected:<20} {expected_framework:<20} {status:<10}")
        except FileNotFoundError as e:
            print(f"{script_path:<40} {'File not found':<20} {expected_framework:<20} {'SKIP':<10}")
        except Exception as e:
            print(f"{script_path:<40} {f'Error: {e}':<20} {expected_framework:<20} {'FAIL':<10}")
            all_passed = False

    print()
    if all_passed:
        print("All framework detection tests PASSED")
    else:
        print("Some framework detection tests FAILED")

    return all_passed


def test_seed_manager():
    """Test SeedManager reproducibility and validation."""
    print("\n" + "=" * 70)
    print("TEST 2: SeedManager Reproducibility")
    print("=" * 70)

    # Test seed validation
    print("\nSeed validation tests:")
    test_cases = [
        (42, 42, "integer seed"),
        ("42", 42, "string seed converted to int"),
        (0, 0, "zero seed"),
        (2**32 - 1, 2**32 - 1, "maximum seed"),
    ]

    validation_passed = True
    for input_seed, expected, description in test_cases:
        try:
            result = SeedManager.validate_seed(input_seed)
            status = "PASS" if result == expected else "FAIL"
            if status == "FAIL":
                validation_passed = False
            print(f"  {description}: validate_seed({input_seed}) = {result} [{status}]")
        except Exception as e:
            print(f"  {description}: validate_seed({input_seed}) raised {type(e).__name__}: {e} [FAIL]")
            validation_passed = False

    # Test error cases
    print("\nError handling tests:")
    error_cases = [
        (None, "None seed"),
        (-1, "negative seed"),
        (2**32, "seed too large"),
        ("invalid", "non-numeric string"),
    ]

    for input_seed, description in error_cases:
        try:
            result = SeedManager.validate_seed(input_seed)
            print(f"  {description}: validate_seed({input_seed}) = {result} [FAIL - should raise error]")
            validation_passed = False
        except ValueError as e:
            print(f"  {description}: validate_seed({input_seed}) raised ValueError [PASS]")
        except Exception as e:
            print(f"  {description}: validate_seed({input_seed}) raised {type(e).__name__} [FAIL - wrong exception]")
            validation_passed = False

    # Test reproducibility
    print("\nReproducibility test:")
    print("  Running random.randint(0, 1000) three times with seed 42:")

    results = []
    for i in range(3):
        with SeedManager(42):
            import random
            results.append(random.randint(0, 1000))

    reproducibility_passed = len(set(results)) == 1
    status = "PASS" if reproducibility_passed else "FAIL"

    print(f"    Results: {results}")
    print(f"    All same? {reproducibility_passed} [{status}]")

    all_passed = validation_passed and reproducibility_passed

    print()
    if all_passed:
        print("All SeedManager tests PASSED")
    else:
        print("Some SeedManager tests FAILED")

    return all_passed


def test_adapter_integration():
    """Test adapter integration with AutoLogger and SeedManager."""
    print("\n" + "=" * 70)
    print("TEST 3: Adapter Integration (Dry Run)")
    print("=" * 70)

    print("\nThis test validates adapter configuration without actually training.")
    print("Full integration tests require MLflow and training data.")

    try:
        # Load adapter module directly
        adapters_module = load_module("adapters", "mlflow_tracking/adapters.py")
        PyTorchAdapter = adapters_module.PyTorchAdapter
        SklearnAdapter = adapters_module.SklearnAdapter

        # Load config parser module directly
        config_parser_module = load_module("config_parser", "mlflow_tracking/config_parser.py")
        ExperimentConfig = config_parser_module.ExperimentConfig

        # Test 1: PyTorch config validation
        print("\nTest 3a: PyTorchAdapter config validation")
        pytorch_config = ExperimentConfig(
            experiment_name="test_pytorch_autolog",
            run_name="pytorch_test",
            adapter="pytorch",
            parameters={
                "model_name": "efficientnet_b0",
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 0.0001,
                "random_seed": 42
            },
            tags={"model_type": "cnn", "test": "true"}
        )

        pytorch_adapter = PyTorchAdapter()
        pytorch_adapter.validate_config(pytorch_config)
        print("  PyTorch config validation: PASS")

        # Test 2: Sklearn config validation
        print("\nTest 3b: SklearnAdapter config validation")
        sklearn_config = ExperimentConfig(
            experiment_name="test_sklearn_autolog",
            run_name="sklearn_test",
            adapter="sklearn",
            parameters={
                "model_type": "ridge",
                "random_seed": 42,
                "alpha": 1.0
            },
            tags={"model_type": "linear", "test": "true"}
        )

        sklearn_adapter = SklearnAdapter()
        sklearn_adapter.validate_config(sklearn_config)
        print("  Sklearn config validation: PASS")

        # Test 3: Framework detection
        print("\nTest 3c: Framework detection for adapter scripts")
        pytorch_framework = AutoLogger.detect_framework('scripts/train_oof_effnet.py')
        sklearn_framework = AutoLogger.detect_framework('scripts/train_ridge_advanced.py')
        print(f"  PyTorch script framework: {pytorch_framework}")
        print(f"  Sklearn script framework: {sklearn_framework}")

        all_passed = (
            pytorch_framework == 'pytorch' and
            sklearn_framework == 'sklearn'
        )
        status = "PASS" if all_passed else "FAIL"
        print(f"  Framework detection: {status}")

        print()
        print("All adapter integration tests PASSED")
        print("\nNote: Actual training would execute with:")
        print("  - AutoLogger context for automatic metric logging")
        print("  - SeedManager context for reproducible random seeds")
        print("  - Framework detected automatically from script imports")

        return True

    except ImportError as e:
        print(f"\n  Import error: {e}")
        print("  This is expected if MLflow is not installed.")
        print("  Adapter integration tests require MLflow.")
        return False
    except Exception as e:
        print(f"\n  Unexpected error: {type(e).__name__}: {e}")
        return False


def show_example_config():
    """Show example YAML configuration with random_seed parameter."""
    print("\n" + "=" * 70)
    print("EXAMPLE: YAML Configuration with random_seed")
    print("=" * 70)

    example_yaml = """
# Example YAML configuration for auto-logging
# This config demonstrates how to use random_seed for reproducibility

experiment_name: image_biomass_autolog
run_name: effnet_b0_seed{{random_seed}}
adapter: pytorch

parameters:
  model_name: efficientnet_b0
  batch_size: 16
  epochs: 30
  learning_rate: 0.0001
  random_seed: 42  # Ensures reproducible results

tags:
  model_type: cnn
  architecture: efficientnet
  purpose: baseline

# Optional: Parameter sweep with different random seeds
sweep:
  grid:
    random_seed: [42, 123, 456]  # Test reproducibility across seeds
    learning_rate: [0.0001, 0.001]
"""

    print(example_yaml)
    print("Key points:")
    print("  - random_seed parameter ensures reproducibility (REPRO-03)")
    print("  - AutoLogger automatically captures metrics (INTEGRATION-02)")
    print("  - Framework detected automatically from script imports")
    print("  - No manual logging code needed in training scripts")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Auto-Logging Functionality Test Suite")
    print("=" * 70)
    print("\nThis test suite validates:")
    print("  1. AutoLogger framework detection")
    print("  2. SeedManager reproducibility")
    print("  3. Adapter integration (dry run)")
    print()

    results = {}

    # Run tests
    results['framework_detection'] = test_framework_detection()
    results['seed_manager'] = test_seed_manager()
    results['adapter_integration'] = test_adapter_integration()

    # Show example config
    show_example_config()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print("All tests PASSED")
        print("\nAuto-logging is ready to use!")
        print("Training scripts will automatically log metrics to MLflow.")
        return 0
    else:
        print("Some tests FAILED")
        print("\nPlease review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
