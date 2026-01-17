"""
Test suite for Optuna hyperparameter optimization.

This module tests the OptimizationConfig schema, OptunaOptimizer class,
and CLI integration for hyperparameter optimization.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import yaml

# Note: These tests don't require optuna to be installed
# They mock the Optuna components to test the integration logic


class TestOptimizationConfig(unittest.TestCase):
    """Test OptimizationConfig schema validation."""

    def test_valid_optimization_config(self):
        """Test creating a valid optimization config."""
        from mlflow_tracking import OptimizationConfig

        config = OptimizationConfig(
            n_trials=100,
            study_name="test_study",
            direction="minimize",
            metric="val.rmse",
            search={
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-1,
                    "log": True
                }
            }
        )

        self.assertEqual(config.n_trials, 100)
        self.assertEqual(config.study_name, "test_study")
        self.assertEqual(config.direction, "minimize")
        self.assertEqual(config.metric, "val.rmse")
        self.assertEqual(len(config.search), 1)

    def test_float_parameter_config(self):
        """Test float parameter search space."""
        from mlflow_tracking import OptimizationConfig

        config = OptimizationConfig(
            n_trials=50,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={
                "lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "alpha": {"type": "float", "low": 0.01, "high": 100.0, "log": False}
            }
        )

        self.assertIn("lr", config.search)
        self.assertIn("alpha", config.search)

    def test_int_parameter_config(self):
        """Test int parameter search space."""
        from mlflow_tracking import OptimizationConfig

        config = OptimizationConfig(
            n_trials=50,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={
                "batch_size": {"type": "int", "low": 8, "high": 64, "step": 8},
                "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1}
            }
        )

        self.assertIn("batch_size", config.search)
        self.assertEqual(config.search["batch_size"]["step"], 8)

    def test_categorical_parameter_config(self):
        """Test categorical parameter search space."""
        from mlflow_tracking import OptimizationConfig

        config = OptimizationConfig(
            n_trials=50,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={
                "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "adamw"]},
                "activation": {"type": "categorical", "choices": ["relu", "gelu"]}
            }
        )

        self.assertIn("optimizer", config.search)
        self.assertEqual(config.search["optimizer"]["choices"], ["adam", "sgd", "adamw"])

    def test_mixed_parameter_types(self):
        """Test config with mixed parameter types."""
        from mlflow_tracking import OptimizationConfig

        config = OptimizationConfig(
            n_trials=100,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={
                "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
                "batch_size": {"type": "int", "low": 8, "high": 64, "step": 8},
                "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}
            }
        )

        self.assertEqual(len(config.search), 3)

    def test_invalid_search_space_empty(self):
        """Test validation fails for empty search space."""
        from mlflow_tracking import OptimizationConfig
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            OptimizationConfig(
                n_trials=100,
                study_name="test",
                direction="minimize",
                metric="val.rmse",
                search={}  # Empty search space
            )

    def test_invalid_pruner_type(self):
        """Test validation fails for invalid pruner type."""
        from mlflow_tracking import OptimizationConfig
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            OptimizationConfig(
                n_trials=100,
                study_name="test",
                direction="minimize",
                metric="val.rmse",
                search={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
                pruner={"type": "invalid_pruner"}
            )

    def test_valid_pruner_configs(self):
        """Test all valid pruner types."""
        from mlflow_tracking import OptimizationConfig

        # Median pruner
        config1 = OptimizationConfig(
            n_trials=100,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
            pruner={"type": "median", "n_startup_trials": 5, "n_warmup_steps": 10}
        )
        self.assertEqual(config1.pruner["type"], "median")

        # Hyperband pruner
        config2 = OptimizationConfig(
            n_trials=100,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
            pruner={"type": "hyperband", "min_resource": 1, "max_resource": 100}
        )
        self.assertEqual(config2.pruner["type"], "hyperband")

        # Successive halving pruner
        config3 = OptimizationConfig(
            n_trials=100,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
            pruner={"type": "successive_halving", "reduction_factor": 4}
        )
        self.assertEqual(config3.pruner["type"], "successive_halving")

    def test_categorical_requires_choices(self):
        """Test validation fails for categorical without choices."""
        from mlflow_tracking import OptimizationConfig, SearchParamConfig
        from pydantic import ValidationError

        # Test at SearchParamConfig level
        with self.assertRaises(ValidationError):
            SearchParamConfig(type="categorical")  # Missing choices


class TestOptunaOptimizer(unittest.TestCase):
    """Test OptunaOptimizer class (with mocked Optuna)."""

    def setUp(self):
        """Set up test fixtures."""
        from mlflow_tracking import OptimizationConfig, ExperimentConfig

        self.optimization_config = OptimizationConfig(
            n_trials=10,
            study_name="test_study",
            direction="minimize",
            metric="val.rmse",
            search={
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}
            }
        )

        self.base_config = ExperimentConfig(
            experiment_name="test_experiment",
            run_name="test_run",
            adapter="pytorch",
            parameters={
                "model_name": "efficientnet_b0",
                "batch_size": 16,
                "epochs": 5
            },
            tags={"test": "true"},
            random_seed=42
        )

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    @patch('mlflow_tracking.optuna_optimizer.AdapterRegistry')
    @patch('mlflow_tracking.optuna_optimizer.ExperimentTracker')
    def test_optimizer_initialization(self, mock_tracker, mock_registry, mock_optuna):
        """Test OptunaOptimizer initialization."""
        from mlflow_tracking import OptunaOptimizer

        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance

        optimizer = OptunaOptimizer(
            optimization_config=self.optimization_config,
            base_config=self.base_config,
            tracker=mock_tracker_instance
        )

        self.assertEqual(optimizer.optimization_config, self.optimization_config)
        self.assertEqual(optimizer.base_config, self.base_config)
        self.assertEqual(optimizer.tracker, mock_tracker_instance)
        self.assertIsNone(optimizer.study)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    @patch('mlflow_tracking.optuna_optimizer.AdapterRegistry')
    @patch('mlflow_tracking.optuna_optimizer.ExperimentTracker')
    def test_search_space_parsing(self, mock_tracker, mock_registry, mock_optuna):
        """Test search space parsing."""
        from mlflow_tracking import OptunaOptimizer

        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance

        optimizer = OptunaOptimizer(
            optimization_config=self.optimization_config,
            base_config=self.base_config,
            tracker=mock_tracker_instance
        )

        self.assertIn("learning_rate", optimizer._search_space)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    @patch('mlflow_tracking.optuna_optimizer.AdapterRegistry')
    @patch('mlflow_tracking.optuna_optimizer.ExperimentTracker')
    def test_pruner_creation_median(self, mock_tracker, mock_registry, mock_optuna):
        """Test median pruner creation."""
        from mlflow_tracking import OptunaOptimizer
        from mlflow_tracking import OptimizationConfig

        config_with_pruner = OptimizationConfig(
            n_trials=10,
            study_name="test",
            direction="minimize",
            metric="val.rmse",
            search={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
            pruner={"type": "median", "n_startup_trials": 5, "n_warmup_steps": 10}
        )

        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance

        optimizer = OptunaOptimizer(
            optimization_config=config_with_pruner,
            base_config=self.base_config,
            tracker=mock_tracker_instance
        )

        # Should not raise an error
        pruner = optimizer._create_pruner()
        self.assertIsNotNone(pruner)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    @patch('mlflow_tracking.optuna_optimizer.AdapterRegistry')
    @patch('mlflow_tracking.optuna_optimizer.ExperimentTracker')
    def test_invalid_optimization_config(self, mock_tracker, mock_registry, mock_optuna):
        """Test error handling for invalid optimization config."""
        from mlflow_tracking import OptunaOptimizer

        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance

        with self.assertRaises(ValueError):
            OptunaOptimizer(
                optimization_config="not_a_config",  # Invalid type
                base_config=self.base_config,
                tracker=mock_tracker_instance
            )


class TestSuggestParamsFromTrial(unittest.TestCase):
    """Test suggest_params_from_trial helper function."""

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    def test_suggest_float_params(self, mock_optuna):
        """Test suggesting float parameters."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        # Mock trial object
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.001

        search_space = {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "alpha": {"type": "float", "low": 0.01, "high": 100.0, "log": False}
        }

        params = suggest_params_from_trial(mock_trial, search_space)

        self.assertIn("lr", params)
        self.assertIn("alpha", params)
        self.assertEqual(mock_trial.suggest_float.call_count, 2)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    def test_suggest_int_params(self, mock_optuna):
        """Test suggesting int parameters."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        # Mock trial object
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 32

        search_space = {
            "batch_size": {"type": "int", "low": 8, "high": 64, "step": 8},
            "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1}
        }

        params = suggest_params_from_trial(mock_trial, search_space)

        self.assertIn("batch_size", params)
        self.assertIn("max_depth", params)
        self.assertEqual(mock_trial.suggest_int.call_count, 2)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    def test_suggest_categorical_params(self, mock_optuna):
        """Test suggesting categorical parameters."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        # Mock trial object
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = "adam"

        search_space = {
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "adamw"]},
            "activation": {"type": "categorical", "choices": ["relu", "gelu"]}
        }

        params = suggest_params_from_trial(mock_trial, search_space)

        self.assertIn("optimizer", params)
        self.assertIn("activation", params)
        self.assertEqual(mock_trial.suggest_categorical.call_count, 2)

    @patch('mlflow_tracking.optuna_optimizer.optuna')
    def test_mixed_parameter_types(self, mock_optuna):
        """Test suggesting mixed parameter types."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        # Mock trial object
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 32
        mock_trial.suggest_categorical.return_value = "adam"

        search_space = {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "int", "low": 8, "high": 64, "step": 8},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}
        }

        params = suggest_params_from_trial(mock_trial, search_space)

        self.assertEqual(len(params), 3)
        self.assertIn("lr", params)
        self.assertIn("batch_size", params)
        self.assertIn("optimizer", params)

    def test_missing_bounds_for_float(self):
        """Test error handling for float without bounds."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        mock_trial = Mock()
        search_space = {
            "lr": {"type": "float"}  # Missing low and high
        }

        with self.assertRaises(ValueError):
            suggest_params_from_trial(mock_trial, search_space)

    def test_missing_choices_for_categorical(self):
        """Test error handling for categorical without choices."""
        from mlflow_tracking.optuna_optimizer import suggest_params_from_trial

        mock_trial = Mock()
        search_space = {
            "optimizer": {"type": "categorical"}  # Missing choices
        }

        with self.assertRaises(ValueError):
            suggest_params_from_trial(mock_trial, search_space)


class TestCreateOptimizationConfig(unittest.TestCase):
    """Test create_optimization_config helper function."""

    def test_create_from_dict(self):
        """Test creating OptimizationConfig from dict."""
        from mlflow_tracking import create_optimization_config

        config_dict = {
            "n_trials": 100,
            "study_name": "test_study",
            "direction": "minimize",
            "metric": "val.rmse",
            "search": {
                "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}
            }
        }

        config = create_optimization_config(config_dict)

        self.assertEqual(config.n_trials, 100)
        self.assertEqual(config.study_name, "test_study")

    def test_invalid_config_dict(self):
        """Test error handling for invalid config dict."""
        from mlflow_tracking import create_optimization_config

        invalid_dict = {
            "n_trials": 100,
            # Missing required fields
        }

        with self.assertRaises(ValueError):
            create_optimization_config(invalid_dict)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration for optimization."""

    def test_cli_imports(self):
        """Test that CLI functions can be imported."""
        try:
            from mlflow_tracking import main_optimize, exp_run_optimize_command
            self.assertTrue(callable(main_optimize))
            self.assertTrue(callable(exp_run_optimize_command))
        except ImportError as e:
            self.fail(f"Failed to import CLI functions: {e}")

    def test_optimization_config_in_experiment_config(self):
        """Test that ExperimentConfig can have optimization section."""
        from mlflow_tracking import ExperimentConfig

        config = ExperimentConfig(
            experiment_name="test",
            run_name="test",
            adapter="pytorch",
            parameters={"lr": 0.001},
            tags={},
            optimization={
                "n_trials": 100,
                "study_name": "test_study",
                "direction": "minimize",
                "metric": "val.rmse",
                "search": {
                    "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}
                }
            }
        )

        self.assertIsNotNone(config.optimization)
        self.assertEqual(config.optimization.n_trials, 100)


class TestExampleConfigs(unittest.TestCase):
    """Test example optimization configurations."""

    def test_effnet_lr_search_config(self):
        """Test that effnet_lr_search config is valid."""
        from mlflow_tracking import ConfigParser, ExperimentConfig

        # This test assumes the example config exists
        config_path = Path("examples/configs/optimization/01_effnet_lr_search.yaml")

        if config_path.exists():
            config = ConfigParser.load_config(config_path)
            self.assertIsInstance(config, ExperimentConfig)
            self.assertIsNotNone(config.optimization)
            self.assertIn("learning_rate", config.optimization.search)

    def test_ridge_alpha_search_config(self):
        """Test that ridge_alpha_search config is valid."""
        from mlflow_tracking import ConfigParser, ExperimentConfig

        config_path = Path("examples/configs/optimization/02_ridge_alpha_search.yaml")

        if config_path.exists():
            config = ConfigParser.load_config(config_path)
            self.assertIsInstance(config, ExperimentConfig)
            self.assertIsNotNone(config.optimization)
            self.assertIn("alpha", config.optimization.search)

    def test_xgboost_multi_param_config(self):
        """Test that xgboost_multi_param config is valid."""
        from mlflow_tracking import ConfigParser, ExperimentConfig

        config_path = Path("examples/configs/optimization/03_xgboost_multi_param.yaml")

        if config_path.exists():
            config = ConfigParser.load_config(config_path)
            self.assertIsInstance(config, ExperimentConfig)
            self.assertIsNotNone(config.optimization)
            # Should have 5 parameters
            self.assertEqual(len(config.optimization.search), 5)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
