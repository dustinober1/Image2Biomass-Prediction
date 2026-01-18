"""
Test suite for InsightsGenerator class.

Tests statistical testing, effect size calculation, insights generation,
hyperparameter correlation analysis, and experiment ranking.
"""

import os
import tempfile
import shutil
from typing import List

import numpy as np
import pandas as pd

from mlflow_tracking.tracker import ExperimentTracker
from mlflow_tracking.analytics.insights_generator import InsightsGenerator


def test_1_insights_generator_initialization():
    """Test InsightsGenerator initialization."""
    print("Test 1: InsightsGenerator initialization")

    generator = InsightsGenerator()

    # Verify comparator is initialized
    assert generator.comparator is not None, "Comparator should be initialized"

    print("  ✓ InsightsGenerator initializes correctly")


def test_2_normality_testing():
    """Test normality testing with normal and non-normal data."""
    print("Test 2: Normality testing")

    generator = InsightsGenerator()

    # Test with normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 50)
    is_normal, p_value = generator._check_normality(normal_data)
    assert is_normal, "Normal data should pass normality test"
    assert 0 <= p_value <= 1, "P-value should be between 0 and 1"

    # Test with non-normal data (exponential)
    non_normal_data = np.random.exponential(1, 50)
    is_normal, p_value = generator._check_normality(non_normal_data)
    assert not is_normal, "Exponential data should fail normality test"

    # Test with small sample (should warn)
    small_data = np.random.normal(0, 1, 15)
    is_normal, p_value = generator._check_normality(small_data)
    # Should still return result, just with warning

    print("  ✓ Normality testing works correctly")


def test_3_statistical_testing_auto_mode():
    """Test statistical testing with automatic test selection."""
    print("Test 3: Statistical testing (auto mode)")

    generator = InsightsGenerator()

    # Create two groups with known difference
    np.random.seed(42)
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(110, 10, 30)

    # Perform test
    result = generator.perform_statistical_test(group1, group2, test_type="auto")

    # Verify result structure
    assert "test_type" in result, "Missing test_type"
    assert "p_value" in result, "Missing p_value"
    assert "effect_size" in result, "Missing effect_size"
    assert "significant" in result, "Missing significant"
    assert "mean_group1" in result, "Missing mean_group1"
    assert "mean_group2" in result, "Missing mean_group2"
    assert "improvement_pct" in result, "Missing improvement_pct"

    # Verify test type is either t-test or Mann-Whitney U
    assert result["test_type"] in ["t-test", "Mann-Whitney U"], f"Invalid test type: {result['test_type']}"

    # Verify means are correct
    assert abs(result["mean_group1"] - 100) < 5, "Mean of group1 should be around 100"
    assert abs(result["mean_group2"] - 110) < 5, "Mean of group2 should be around 110"

    print(f"  ✓ Statistical testing works (test: {result['test_type']}, p-value: {result['p_value']:.4f})")


def test_4_statistical_testing_manual_mode():
    """Test statistical testing with manual test selection."""
    print("Test 4: Statistical testing (manual mode)")

    generator = InsightsGenerator()

    # Create two groups
    np.random.seed(42)
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(105, 10, 30)

    # Test with t-test
    result_ttest = generator.perform_statistical_test(group1, group2, test_type="ttest")
    assert result_ttest["test_type"] == "t-test", "Should use t-test"

    # Test with Mann-Whitney U
    result_mw = generator.perform_statistical_test(group1, group2, test_type="mannwhitney")
    assert result_mw["test_type"] == "Mann-Whitney U", "Should use Mann-Whitney U"

    print("  ✓ Manual test selection works correctly")


def test_5_effect_size_calculation():
    """Test effect size calculation."""
    print("Test 5: Effect size calculation")

    generator = InsightsGenerator()

    # Test with groups of known difference
    group1 = np.array([100, 102, 98, 101, 99])
    group2 = np.array([110, 112, 108, 111, 109])

    cohens_d = generator._calculate_effect_size(group1, group2)
    assert isinstance(cohens_d, float), "Cohen's d should be float"
    assert abs(cohens_d) > 2.0, "Effect size should be large for this data"

    # Test with identical groups (should return 0)
    group_identical = np.array([100, 100, 100, 100, 100])
    cohens_d_identical = generator._calculate_effect_size(group_identical, group_identical)
    assert cohens_d_identical == 0.0, "Effect size should be 0 for identical groups"

    print("  ✓ Effect size calculation works correctly")


def test_6_effect_size_interpretation():
    """Test effect size interpretation."""
    print("Test 6: Effect size interpretation")

    generator = InsightsGenerator()

    # Test different effect sizes
    assert generator._interpret_effect_size(0.1) == "negligible", "0.1 should be negligible"
    assert generator._interpret_effect_size(0.3) == "small", "0.3 should be small"
    assert generator._interpret_effect_size(0.6) == "medium", "0.6 should be medium"
    assert generator._interpret_effect_size(1.0) == "large", "1.0 should be large"

    # Test negative values (should use absolute value)
    assert generator._interpret_effect_size(-0.3) == "small", "-0.3 should be small"
    assert generator._interpret_effect_size(-1.0) == "large", "-1.0 should be large"

    print("  ✓ Effect size interpretation works correctly")


def test_7_recommendation_generation():
    """Test recommendation generation for different scenarios."""
    print("Test 7: Recommendation generation")

    generator = InsightsGenerator()

    # Test 1: Not significant (p_value >= 0.05)
    rec1 = generator._generate_recommendation(0.10, 0.5, 5.0)
    assert "No significant difference" in rec1, f"Expected 'No significant difference', got: {rec1}"

    # Test 2: Significant but negligible (p_value < 0.05, effect_size < 0.2)
    rec2 = generator._generate_recommendation(0.01, 0.1, 2.0)
    assert "negligible effect" in rec2, f"Expected 'negligible effect', got: {rec2}"

    # Test 3: Significant small effect (0.2 <= effect_size < 0.5)
    rec3 = generator._generate_recommendation(0.01, 0.3, 5.0)
    assert "small effect" in rec3, f"Expected 'small effect', got: {rec3}"

    # Test 4: Significant medium effect (0.5 <= effect_size < 0.8)
    rec4 = generator._generate_recommendation(0.01, 0.6, 10.0)
    assert "medium effect" in rec4, f"Expected 'medium effect', got: {rec4}"

    # Test 5: Significant large effect (effect_size >= 0.8)
    rec5 = generator._generate_recommendation(0.01, 1.2, 15.0)
    assert "large effect" in rec5, f"Expected 'large effect', got: {rec5}"
    assert "15.0%" in rec5, f"Expected percentage in recommendation, got: {rec5}"

    print("  ✓ Recommendation generation works correctly")


def test_8_insights_generation_insufficient_data():
    """Test insights generation with insufficient data."""
    print("Test 8: Insights generation (insufficient data)")

    generator = InsightsGenerator()

    # Test with too few runs
    result = generator.generate_insights(["run1", "run2"], min_sample_size=5)

    assert result["status"] == "insufficient_data", "Expected insufficient_data status"
    assert "recommendation" in result, "Missing recommendation"
    assert result["sample_size"] == 2, "Sample size should be 2"

    print("  ✓ Insufficient data handling works correctly")


def test_9_insights_generation_success():
    """Test insights generation with MLflow data."""
    print("Test 9: Insights generation (success)")

    # Create temporary directory for MLflow
    temp_dir = tempfile.mkdtemp()
    tracking_uri = f"file://{temp_dir}/mlflow"

    try:
        # Initialize tracker
        tracker = ExperimentTracker("test_insights", tracking_uri=tracking_uri)
        generator = InsightsGenerator(tracking_uri=tracking_uri)

        # Log 10 runs with different metrics
        run_ids = []
        np.random.seed(42)

        for i in range(10):
            tracker.start_run(f"run_{i}")

            # Log parameters
            tracker.log_params({"learning_rate": 0.001 * (i + 1), "batch_size": 32})

            # Log metrics (with some variation)
            base_rmse = 10.0 - i * 0.5  # Improving with learning rate
            tracker.log_metrics({
                "train.rmse": base_rmse + np.random.normal(0, 0.5),
                "val.rmse": base_rmse + np.random.normal(0, 0.3),
                "val.mae": base_rmse * 0.8 + np.random.normal(0, 0.2),
            })

            run_ids.append(tracker.get_run_id())
            tracker.end_run()

        # Generate insights
        result = generator.generate_insights(run_ids, metric="val.rmse", min_sample_size=5)

        # Verify result structure
        assert result["status"] == "success", "Expected success status"
        assert "best_run" in result, "Missing best_run"
        assert "best_metric" in result, "Missing best_metric"
        assert "summary_statistics" in result, "Missing summary_statistics"
        assert "statistical_tests" in result, "Missing statistical_tests"
        assert "recommendation" in result, "Missing recommendation"
        assert result["sample_size"] == 10, "Sample size should be 10"

        # Verify summary statistics
        stats = result["summary_statistics"]
        assert "mean" in stats, "Missing mean"
        assert "std" in stats, "Missing std"
        assert "min" in stats, "Missing min"
        assert "max" in stats, "Missing max"
        assert "range" in stats, "Missing range"

        print(f"  ✓ Insights generation works (best metric: {result['best_metric']:.2f})")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_10_hyperparameter_correlation_analysis():
    """Test hyperparameter correlation analysis."""
    print("Test 10: Hyperparameter correlation analysis")

    # Create temporary directory for MLflow
    temp_dir = tempfile.mkdtemp()
    tracking_uri = f"file://{temp_dir}/mlflow"

    try:
        # Initialize tracker
        tracker = ExperimentTracker("test_correlation", tracking_uri=tracking_uri)
        generator = InsightsGenerator(tracking_uri=tracking_uri)

        # Log runs with different hyperparameters
        run_ids = []
        np.random.seed(42)

        for i in range(20):
            tracker.start_run(f"run_{i}")

            # Log parameters (with correlation to metric)
            lr = 0.001 * (i + 1)
            tracker.log_params({
                "learning_rate": lr,
                "batch_size": 32 if i % 2 == 0 else 64,
            })

            # Log metrics (higher learning rate -> lower RMSE)
            rmse = 10.0 - lr * 100 + np.random.normal(0, 0.5)
            tracker.log_metrics({"val.rmse": rmse})

            run_ids.append(tracker.get_run_id())
            tracker.end_run()

        # Analyze hyperparameters
        corr_df = generator.compare_hyperparameters(run_ids, metric="val.rmse", top_n=10)

        # Verify DataFrame structure
        assert "parameter" in corr_df.columns, "Missing parameter column"
        assert "correlation" in corr_df.columns, "Missing correlation column"
        assert "abs_correlation" in corr_df.columns, "Missing abs_correlation column"
        assert "direction" in corr_df.columns, "Missing direction column"
        assert "interpretation" in corr_df.columns, "Missing interpretation column"

        # Verify sorting by absolute correlation
        if len(corr_df) > 1:
            abs_corrs = corr_df["abs_correlation"].values
            assert all(abs_corrs[i] >= abs_corrs[i + 1] for i in range(len(abs_corrs) - 1)), \
                "Should be sorted by abs_correlation (descending)"

        print(f"  ✓ Hyperparameter correlation analysis works ({len(corr_df)} parameters analyzed)")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_11_experiment_ranking():
    """Test experiment ranking by multiple metrics."""
    print("Test 11: Experiment ranking")

    # Create temporary directory for MLflow
    temp_dir = tempfile.mkdtemp()
    tracking_uri = f"file://{temp_dir}/mlflow"

    try:
        # Initialize tracker
        tracker = ExperimentTracker("test_ranking", tracking_uri=tracking_uri)
        generator = InsightsGenerator(tracking_uri=tracking_uri)

        # Log runs with multiple metrics
        run_ids = []
        np.random.seed(42)

        for i in range(10):
            tracker.start_run(f"run_{i}")

            # Log metrics (some runs better at RMSE, others at MAE)
            rmse = 10.0 - i * 0.3 + np.random.normal(0, 0.5)
            mae = 8.0 - i * 0.2 + np.random.normal(0, 0.3)
            tracker.log_metrics({
                "val.rmse": rmse,
                "val.mae": mae,
            })

            run_ids.append(tracker.get_run_id())
            tracker.end_run()

        # Rank experiments
        ranked_df = generator.rank_experiments(run_ids, metrics=["val.rmse", "val.mae"])

        # Verify DataFrame structure
        assert "run_id" in ranked_df.columns, "Missing run_id column"
        assert "score" in ranked_df.columns, "Missing score column"
        assert "rank" in ranked_df.columns, "Missing rank column"
        assert "metrics" in ranked_df.columns, "Missing metrics column"
        assert "normalized_metrics" in ranked_df.columns, "Missing normalized_metrics column"

        # Verify sorting by score (descending)
        scores = ranked_df["score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
            "Should be sorted by score (descending)"

        # Verify ranks are sequential
        ranks = ranked_df["rank"].values
        assert list(ranks) == list(range(1, len(ranks) + 1)), "Ranks should be sequential"

        print(f"  ✓ Experiment ranking works ({len(ranked_df)} experiments ranked)")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("InsightsGenerator Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_1_insights_generator_initialization,
        test_2_normality_testing,
        test_3_statistical_testing_auto_mode,
        test_4_statistical_testing_manual_mode,
        test_5_effect_size_calculation,
        test_6_effect_size_interpretation,
        test_7_recommendation_generation,
        test_8_insights_generation_insufficient_data,
        test_9_insights_generation_success,
        test_10_hyperparameter_correlation_analysis,
        test_11_experiment_ranking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all_tests() else 1)
