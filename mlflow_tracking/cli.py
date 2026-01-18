"""
Command-line interface for running experiments from YAML configuration.

This module provides the `exp-run` command that loads YAML configurations,
validates them against registered adapters, and executes experiments using
MLflow for logging.

Also provides `exp-run-batch` command for parallel batch execution.

Also provides `exp-run-optimize` command for hyperparameter optimization.
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, List

# Lazy imports to avoid circular dependency with __init__.py
# These are imported within functions when needed


def exp_run_command(
    config_path: str,
    sweep: bool = False,
    verbose: bool = False
) -> int:
    """Execute experiment from YAML configuration.

    Args:
        config_path: Path to YAML configuration file
        sweep: If True, expand parameter sweeps and run all combinations
        verbose: If True, print detailed execution info

    Returns:
        0 on success, 1 on failure
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import ConfigParser, ExperimentTracker
    from mlflow_tracking.adapters import AdapterRegistry

    try:
        # Load config(s)
        if sweep:
            if verbose:
                print(f"Expanding parameter sweeps from {config_path}...")
            configs = ConfigParser.expand_sweeps(config_path)
            if verbose:
                print(f"Expanded into {len(configs)} configurations")
        else:
            configs = [ConfigParser.load_config(config_path)]

        # Execute each config
        for i, config in enumerate(configs, 1):
            if verbose:
                print(f"\n[{i}/{len(configs)}] Running: {config.experiment_name}/{config.run_name}")

            # Validate config against adapter requirements
            ConfigParser.validate(config, AdapterRegistry)

            # Create experiment tracker
            with ExperimentTracker(
                config.experiment_name,
                auto_log_environment=True
            ) as tracker:
                # Start run
                run_id = tracker.start_run(
                    config.run_name,
                    tags=config.tags,
                    random_seed=config.random_seed
                )

                if verbose:
                    print(f"  Run ID: {run_id}")
                    print(f"  Parameters: {config.parameters}")

                # Log parameters
                tracker.log_params(config.parameters)

                # Get adapter and execute with error handling
                adapter = AdapterRegistry.get(config.adapter)

                try:
                    metrics = adapter.execute(config, tracker)
                    # Log metrics returned by adapter
                    tracker.log_metrics(metrics)

                    if verbose:
                        print(f"  Metrics: {metrics}")

                except Exception as adapter_error:
                    # Mark run as failed in MLflow
                    tracker.mark_failed(str(adapter_error))
                    raise

        if verbose:
            print(f"\nCompleted {len(configs)} experiment(s)")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-run command.

    Usage:
        exp-run config.yaml                    # Run single experiment
        exp-run config.yaml --sweep            # Run all sweep combinations
        exp-run config.yaml --verbose          # Print detailed info
    """
    parser = argparse.ArgumentParser(
        description="Run experiments from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-run config.yaml                    Run single experiment
  exp-run config.yaml --sweep            Run parameter sweep
  exp-run config.yaml --verbose          Show detailed output
  exp-run config.yaml -s -v              Short flags

For more information, see: examples/configs/README.md
        """
    )

    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "-s", "--sweep",
        action="store_true",
        help="Expand parameter sweeps and run all combinations"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed execution information"
    )

    args = parser.parse_args(argv)

    return exp_run_command(
        config_path=args.config,
        sweep=args.sweep,
        verbose=args.verbose
    )


def exp_run_batch_command(
    dir_path: Optional[str] = None,
    configs: Optional[str] = None,
    max_workers: Optional[int] = None,
    verbose: bool = False
) -> int:
    """Execute batch of experiments from directory or config list.

    Args:
        dir_path: Path to directory containing YAML configs
        configs: Comma-separated list of config file paths
        max_workers: Maximum concurrent experiments (auto-suggest if None)
        verbose: Print detailed execution info

    Returns:
        0 on success, 1 on partial failure, 2 on batch error, 3 on resource error
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import BatchExecutor, ResourceManager

    try:
        # Validate inputs
        if not dir_path and not configs:
            print("Error: Must specify --dir or --configs", file=sys.stderr)
            return 2

        if dir_path and configs:
            print("Error: Cannot specify both --dir and --configs", file=sys.stderr)
            return 2

        # Create executor and resource manager
        executor = BatchExecutor()
        resource_manager = ResourceManager()

        # Load configs
        if dir_path:
            if verbose:
                print(f"Loading configs from directory: {dir_path}")
            configs_list = executor.load_configs_from_dir(dir_path)
        else:
            # Parse comma-separated config paths
            config_paths = [p.strip() for p in configs.split(',')]
            if verbose:
                print(f"Loading {len(config_paths)} config(s)")
            configs_list = executor.load_configs(config_paths)

        if not configs_list:
            print("Error: No valid configurations found", file=sys.stderr)
            return 2

        # Print resource summary
        if verbose:
            summary = resource_manager.get_resource_summary()
            print(f"\nAvailable Resources:")
            print(f"  GPUs: {summary['available_gpus']}/{summary['total_gpus']}")
            print(f"  CPUs: {summary['available_cpus']}/{summary['total_cpus']} ({summary['reserved_cpus']} reserved)")
            print(f"Max concurrent experiments: {max_workers or summary['suggested_concurrent']}")
            print()

        # Execute batch
        results = executor.execute_batch(
            configs_list,
            max_workers=max_workers,
            verbose=verbose
        )

        # Count results
        completed = sum(1 for r in results if r.status == 'completed')
        failed = sum(1 for r in results if r.status == 'failed')

        # Print summary
        if verbose:
            print(f"\nBatch complete: {completed}/{len(results)} succeeded, {failed} failed")

            # Find best result
            best = executor.get_best_result(results, metric='val.rmse', minimize=True)
            if best:
                print(f"\nBest result:")
                print(f"  Run: {best.config.run_name}")
                print(f"  val.rmse: {best.metrics.get('val.rmse', 'N/A')}")
            else:
                print("\nNo successful experiments to compare")

        # Return exit code
        if failed == 0:
            return 0
        elif completed > 0:
            return 1  # Partial failure
        else:
            return 2  # All failed

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 2


def main_batch(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-run-batch command.

    Usage:
        exp-run-batch --dir examples/configs/batch/    # Run all configs in directory
        exp-run-batch --configs exp1.yaml,exp2.yaml    # Run specific configs
        exp-run-batch --dir batch/ --max-workers 4     # Limit concurrency
        exp-run-batch --dir batch/ --verbose            # Detailed output
    """
    parser = argparse.ArgumentParser(
        description="Run batch of experiments from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-run-batch --dir examples/configs/batch/  Run all configs in directory
  exp-run-batch --configs exp1.yaml,exp2.yaml  Run specific configs
  exp-run-batch --dir batch/ --max-workers 4   Limit concurrency
  exp-run-batch --dir batch/ -v               Short flag for verbose

For more information, see: examples/configs/README.md
        """
    )

    # Mutually exclusive group for dir or configs
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dir",
        help="Path to directory containing YAML config files"
    )
    source_group.add_argument(
        "--configs",
        help="Comma-separated list of YAML config file paths"
    )

    parser.add_argument(
        "-m", "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent experiments (auto-suggest if not provided)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed execution information"
    )

    args = parser.parse_args(argv)

    return exp_run_batch_command(
        dir_path=args.dir,
        configs=args.configs,
        max_workers=args.max_workers,
        verbose=args.verbose
    )


def exp_run_optimize_command(
    config_path: str,
    n_trials: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> int:
    """Execute hyperparameter optimization from YAML configuration.

    Args:
        config_path: Path to YAML configuration file with optimization section
        n_trials: Override number of trials (uses config value if None)
        n_jobs: Number of parallel trials (-1 for auto-detect)
        verbose: Print detailed optimization progress

    Returns:
        0 on success, 1 on optimization error, 2 on config error
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import ConfigParser, ExperimentTracker, OptunaOptimizer
    from mlflow_tracking.adapters import AdapterRegistry

    try:
        # Load config
        if verbose:
            print(f"Loading optimization config from {config_path}...")

        config = ConfigParser.load_config(config_path)

        # Check for optimization section
        if config.optimization is None:
            print(
                f"Error: Config file must have 'optimization' section. "
                f"See examples/configs/optimization/ for examples.",
                file=sys.stderr
            )
            return 2

        # Override n_trials if specified
        if n_trials is not None:
            config.optimization.n_trials = n_trials
            if verbose:
                print(f"Override: n_trials = {n_trials}")

        # Validate config against adapter requirements
        if verbose:
            print(f"Validating config for adapter '{config.adapter}'...")
        ConfigParser.validate(config, AdapterRegistry)

        # Create experiment tracker
        tracker = ExperimentTracker(
            config.experiment_name,
            auto_log_environment=True
        )

        # Create optimizer
        optimizer = OptunaOptimizer(
            optimization_config=config.optimization,
            base_config=config,
            tracker=tracker,
        )

        # Handle n_jobs=-1 for auto-detect
        if n_jobs == -1:
            from mlflow_tracking.resource_manager import ResourceManager
            resource_manager = ResourceManager()
            n_jobs = resource_manager.get_resource_summary()['suggested_concurrent']
            if verbose:
                print(f"Auto-detected parallel jobs: {n_jobs}")

        # Run optimization study
        if verbose:
            print(f"\nStarting optimization study...")
            print(f"  Study name: {config.optimization.study_name}")
            print(f"  Trials: {config.optimization.n_trials}")
            print(f"  Direction: {config.optimization.direction}")
            print(f"  Metric: {config.optimization.metric}")
            print(f"  Search space: {list(config.optimization.search.keys())}")
            print(f"  Parallel jobs: {n_jobs}")
            print()

        study = optimizer.run_study(n_jobs=n_jobs, show_progress=verbose)

        # Print results
        print(f"\n{'='*60}")
        print(f"Optimization complete!")
        print(f"{'='*60}")
        print(f"Best value: {study.best_value}")
        print(f"Best params:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        print()

        # Save best config
        best_config = optimizer.generate_best_config()
        best_config_path = Path(config_path).parent / f"{Path(config_path).stem}_best.yaml"

        # Convert config to dict for YAML export
        best_config_dict = best_config.model_dump()

        # Add optimization metadata
        best_config_dict['tags']['optimized_from'] = config.optimization.study_name
        best_config_dict['tags']['best_value'] = str(study.best_value)

        with open(best_config_path, 'w') as f:
            yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Best config saved to: {best_config_path}")
        print(f"\nTo run with best hyperparameters:")
        print(f"  exp-run {best_config_path}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main_optimize(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-run-optimize command.

    Usage:
        exp-run-optimize config.yaml                      # Run optimization
        exp-run-optimize config.yaml --n-trials 50        # Override trial count
        exp-run-optimize config.yaml --n-jobs 4           # Parallel trials
        exp-run-optimize config.yaml --verbose            # Detailed output
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-run-optimize config.yaml                  Run optimization study
  exp-run-optimize config.yaml --n-trials 50    Override trial count
  exp-run-optimize config.yaml --n-jobs 4       Run 4 parallel trials
  exp-run-optimize config.yaml -v               Verbose output
  exp-run-optimize config.yaml --n-jobs -1      Auto-detect parallel jobs

For more information, see: examples/configs/optimization/README.md
        """
    )

    parser.add_argument(
        "config",
        help="Path to YAML configuration file with optimization section"
    )

    parser.add_argument(
        "-n", "--n-trials",
        type=int,
        default=None,
        help="Override number of optimization trials"
    )

    parser.add_argument(
        "-j", "--n-jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel trials (-1 for auto-detect, default: 1)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed optimization progress"
    )

    args = parser.parse_args(argv)

    return exp_run_optimize_command(
        config_path=args.config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


def exp_analyze_errors_command(
    run_id: str,
    predictions_path: str = "predictions.csv",
    output_dir: str = "error_analysis",
    log_artifacts: bool = True,
    verbose: bool = False,
) -> int:
    """Analyze prediction errors and identify failure modes.

    Args:
        run_id: MLflow run ID to analyze
        predictions_path: Path to predictions artifact (default: predictions.csv)
        output_dir: Output directory for plots (default: error_analysis)
        log_artifacts: Log analysis as MLflow artifacts (default: True)
        verbose: Print detailed output

    Returns:
        0 on success, 1 on error
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import ExperimentTracker
    from mlflow_tracking.analytics import ErrorAnalyzer
    from mlflow_tracking.analytics.visualizations import (
        plot_residuals,
        plot_error_distribution,
        plot_prediction_vs_actual,
    )
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    try:
        # Initialize analyzer
        analyzer = ErrorAnalyzer()

        if verbose:
            print(f"Loading predictions from run {run_id}...")

        # Load run and predictions
        analyzer.load_run(run_id, predictions_path)

        # Compute residuals if not already computed
        if analyzer.residuals is None:
            analyzer.compute_residuals()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Generating error analysis plots...")

        # Generate plots
        residual_fig = plot_residuals(analyzer.predictions_df)
        distribution_fig = plot_error_distribution(analyzer.predictions_df)
        pred_vs_actual_fig = plot_prediction_vs_actual(analyzer.predictions_df)

        # Save plots
        residual_path = output_path / "residuals.png"
        distribution_path = output_path / "error_distribution.png"
        pred_vs_actual_path = output_path / "prediction_vs_actual.png"

        residual_fig.savefig(residual_path, dpi=150, bbox_inches='tight')
        distribution_fig.savefig(distribution_path, dpi=150, bbox_inches='tight')
        pred_vs_actual_fig.savefig(pred_vs_actual_path, dpi=150, bbox_inches='tight')

        plt.close('all')

        # Identify failure modes
        if verbose:
            print(f"Identifying failure modes...")

        failure_modes = analyzer.identify_failure_modes(n_clusters=3)

        # Get error statistics
        stats = analyzer.get_error_statistics()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Error Analysis Summary")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"\nError Statistics:")
        print(f"  Mean Absolute Error: {stats['mean_abs_error']:.4f}")
        print(f"  Median Absolute Error: {stats['median_abs_error']:.4f}")
        print(f"  Max Absolute Error: {stats['max_abs_error']:.4f}")
        print(f"  Std Residual: {stats['std_residual']:.4f}")
        print(f"  90th Percentile: {stats['percentiles']['90']:.4f}")
        print(f"  95th Percentile: {stats['percentiles']['95']:.4f}")
        print(f"  99th Percentile: {stats['percentiles']['99']:.4f}")

        if failure_modes is not None:
            print(f"\nFailure Modes:")
            for cluster_id in sorted(failure_modes['cluster'].unique()):
                cluster_data = failure_modes[failure_modes['cluster'] == cluster_id]
                print(f"  Cluster {cluster_id}:")
                print(f"    Samples: {len(cluster_data)}")
                print(f"    Mean Abs Error: {cluster_data['abs_residual'].mean():.4f}")
                print(f"    Mean Pct Error: {cluster_data['pct_error'].mean():.2f}%")

        print(f"\nPlots saved to:")
        print(f"  {residual_path}")
        print(f"  {distribution_path}")
        print(f"  {pred_vs_actual_path}")

        # Log artifacts if requested
        if log_artifacts:
            if verbose:
                print(f"\nLogging analysis to MLflow...")

            # Create child run for error analysis
            with ExperimentTracker(
                "error_analysis",
                auto_log_environment=False
            ) as tracker:
                analysis_run_id = tracker.start_run(
                    f"error_analysis_{run_id[:8]}",
                    tags={"parent_run_id": run_id}
                )

                # Log plots as artifacts
                tracker.log_artifacts(str(output_path))

                # Log error statistics as metrics
                tracker.log_metrics({
                    "mean_abs_error": stats['mean_abs_error'],
                    "median_abs_error": stats['median_abs_error'],
                    "max_abs_error": stats['max_abs_error'],
                    "std_residual": stats['std_residual'],
                    "p90_abs_error": stats['percentiles']['90'],
                    "p95_abs_error": stats['percentiles']['95'],
                    "p99_abs_error": stats['percentiles']['99'],
                })

                if verbose:
                    print(f"Error analysis logged to run: {analysis_run_id}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error analysis error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main_analyze_errors(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-analyze-errors command.

    Usage:
        exp-analyze-errors <run_id>                             # Analyze errors
        exp-analyze-errors <run_id> --output-dir results/      # Custom output
        exp-analyze-errors <run_id> --no-log-artifacts         # Skip MLflow logging
        exp-analyze-errors <run_id> --verbose                  # Detailed output
    """
    parser = argparse.ArgumentParser(
        description="Analyze prediction errors and identify failure modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-analyze-errors abc123def456                    Analyze errors for run
  exp-analyze-errors abc123def456 --output-dir err/  Custom output directory
  exp-analyze-errors abc123def456 --no-log-artifacts Skip MLflow logging
  exp-analyze-errors abc123def456 -v                 Verbose output

For more information, see: examples/configs/analytics/README.md
        """
    )

    parser.add_argument(
        "run_id",
        help="MLflow run ID to analyze"
    )

    parser.add_argument(
        "--predictions-path",
        default="predictions.csv",
        help="Path to predictions artifact within run (default: predictions.csv)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="error_analysis",
        help="Output directory for plots (default: error_analysis)"
    )

    parser.add_argument(
        "--no-log-artifacts",
        action="store_false",
        dest="log_artifacts",
        help="Don't log analysis as MLflow artifacts"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed analysis information"
    )

    args = parser.parse_args(argv)

    return exp_analyze_errors_command(
        run_id=args.run_id,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        log_artifacts=args.log_artifacts,
        verbose=args.verbose,
    )


def exp_interpret_command(
    run_id: str,
    output_dir: str = "interpretability",
    plot_type: str = "summary",
    max_features: int = 20,
    compute_permutation: bool = False,
    log_artifacts: bool = True,
    verbose: bool = False,
) -> int:
    """Generate model interpretability analysis (SHAP/ELI5).

    Args:
        run_id: MLflow run ID to analyze
        output_dir: Output directory for plots (default: interpretability)
        plot_type: Type of SHAP plot (summary, bar, dependence) (default: summary)
        max_features: Max features to show (default: 20)
        compute_permutation: Also compute permutation importance (default: False)
        log_artifacts: Log analysis as MLflow artifacts (default: True)
        verbose: Print detailed output

    Returns:
        0 on success, 1 on error
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import ExperimentTracker
    from mlflow_tracking.analytics import ModelInterpretability
    from mlflow_tracking.analytics.visualizations import (
        plot_feature_importance,
        plot_local_explanation,
    )
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        # Initialize interpreter
        interpreter = ModelInterpretability()

        if verbose:
            print(f"Loading model from run {run_id}...")

        # Load model from artifacts
        model = interpreter._load_model_from_artifacts(run_id)

        # Load test data (try to get from run artifacts or ask user)
        # For now, we'll create a placeholder for the user to provide data
        print("Note: Test data required for SHAP analysis.")
        print("Please provide a path to test data CSV with --test-data-path.")
        print("The CSV should have the same features as the training data.")

        # This is a placeholder - in real usage, user would provide test data
        # For now, we'll return an error message
        print("\nError: Test data required for interpretability analysis.", file=sys.stderr)
        print("Use --test-data-path to specify the test data file.", file=sys.stderr)
        return 1

        # The rest of this function would execute if test data was provided:
        # X_test = load_test_data(...)
        # shap_values, explainer = interpreter.compute_shap(run_id, X_test)
        # ... generate plots ...
        # ... save plots ...
        # ... log artifacts ...

    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print("Please install SHAP: pip install shap", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Interpretability error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main_interpret(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-interpret command.

    Usage:
        exp-interpret <run_id>                                # Generate interpretability
        exp-interpret <run_id> --plot-type bar               # Bar plot instead
        exp-interpret <run_id> --max-features 30             # Show more features
        exp-interpret <run_id> --compute-permutation         # Add permutation importance
        exp-interpret <run_id> --verbose                     # Detailed output
    """
    parser = argparse.ArgumentParser(
        description="Generate model interpretability analysis (SHAP/ELI5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-interpret abc123def456                       Generate SHAP analysis
  exp-interpret abc123def456 --plot-type bar       Use bar plot
  exp-interpret abc123def456 --max-features 30     Show top 30 features
  exp-interpret abc123def456 --compute-permutation Add permutation importance
  exp-interpret abc123def456 -v                    Verbose output

For more information, see: examples/configs/analytics/README.md
        """
    )

    parser.add_argument(
        "run_id",
        help="MLflow run ID to analyze"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="interpretability",
        help="Output directory for plots (default: interpretability)"
    )

    parser.add_argument(
        "--plot-type",
        default="summary",
        choices=["summary", "bar", "dependence"],
        help="Type of SHAP plot (default: summary)"
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=20,
        help="Maximum number of features to show (default: 20)"
    )

    parser.add_argument(
        "--compute-permutation",
        action="store_true",
        help="Also compute permutation importance"
    )

    parser.add_argument(
        "--no-log-artifacts",
        action="store_false",
        dest="log_artifacts",
        help="Don't log analysis as MLflow artifacts"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed analysis information"
    )

    args = parser.parse_args(argv)

    return exp_interpret_command(
        run_id=args.run_id,
        output_dir=args.output_dir,
        plot_type=args.plot_type,
        max_features=args.max_features,
        compute_permutation=args.compute_permutation,
        log_artifacts=args.log_artifacts,
        verbose=args.verbose,
    )


def exp_insights_command(
    run_ids: str,
    metric: str = "val.rmse",
    group_by: Optional[str] = None,
    output_dir: str = "insights",
    min_sample_size: int = 5,
    log_artifacts: bool = True,
    verbose: bool = False,
) -> int:
    """Generate automated insights from experiment comparisons.

    Args:
        run_ids: Comma-separated list of MLflow run IDs
        metric: Metric to analyze (default: val.rmse)
        group_by: Parameter to group by (e.g., params.learning_rate)
        output_dir: Output directory for reports (default: insights)
        min_sample_size: Minimum runs for statistical testing (default: 5)
        log_artifacts: Log insights as MLflow artifacts (default: True)
        verbose: Print detailed output

    Returns:
        0 on success, 1 on error
    """
    # Lazy imports to avoid circular dependency
    from mlflow_tracking import ExperimentTracker
    from mlflow_tracking.analytics import InsightsGenerator
    from pathlib import Path
    import json

    try:
        # Parse run_ids
        run_ids_list = [r.strip() for r in run_ids.split(",")]

        if verbose:
            print(f"Analyzing {len(run_ids_list)} runs...")

        # Initialize generator
        generator = InsightsGenerator()

        # Generate insights
        insights = generator.generate_insights(
            run_ids_list,
            metric=metric,
            group_by=group_by,
            min_sample_size=min_sample_size
        )

        # Check for insufficient data
        if insights.get("status") == "insufficient_data":
            print(f"Warning: {insights.get('message')}", file=sys.stderr)
            print(f"Recommendation: {insights.get('recommendation')}", file=sys.stderr)
            return 1

        # Compute hyperparameter correlations
        if verbose:
            print(f"Computing hyperparameter correlations...")

        correlations = generator.compare_hyperparameters(run_ids_list, metric=metric)

        # Rank experiments
        if verbose:
            print(f"Ranking experiments...")

        rankings = generator.rank_experiments(run_ids_list)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        insights_path = output_path / "insights.json"
        correlations_path = output_path / "correlations.csv"
        rankings_path = output_path / "rankings.csv"

        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)

        if correlations is not None:
            correlations.to_csv(correlations_path, index=False)

        if rankings is not None:
            rankings.to_csv(rankings_path, index=False)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Automated Insights Summary")
        print(f"{'='*60}")
        print(f"Analyzed {len(run_ids_list)} runs")
        print(f"Metric: {metric}")

        if "best_run" in insights:
            print(f"\nBest Run:")
            print(f"  Run ID: {insights['best_run']}")

        if "summary_statistics" in insights:
            stats = insights["summary_statistics"]
            print(f"\nSummary Statistics:")
            print(f"  Mean: {stats.get('mean', 'N/A')}")
            print(f"  Std: {stats.get('std', 'N/A')}")
            print(f"  Min: {stats.get('min', 'N/A')}")
            print(f"  Max: {stats.get('max', 'N/A')}")

        if "statistical_tests" in insights:
            tests = insights["statistical_tests"]
            if tests:
                print(f"\nStatistical Tests:")
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict):
                        p_value = test_result.get('p_value', 'N/A')
                        effect_size = test_result.get('effect_size', 'N/A')
                        print(f"  {test_name}:")
                        print(f"    p-value: {p_value}")
                        print(f"    effect size: {effect_size}")

        if "recommendation" in insights:
            print(f"\nRecommendation:")
            print(f"  {insights['recommendation']}")

        if correlations is not None and not correlations.empty:
            print(f"\nTop Hyperparameter Correlations:")
            for _, row in correlations.head(5).iterrows():
                print(f"  {row['parameter']}: {row['correlation']:.3f}")

        if rankings is not None and not rankings.empty:
            print(f"\nTop Experiments:")
            for _, row in rankings.head(5).iterrows():
                print(f"  {row['run_id']}: {row.get('score', 'N/A')}")

        print(f"\nResults saved to:")
        print(f"  {insights_path}")
        if correlations is not None:
            print(f"  {correlations_path}")
        if rankings is not None:
            print(f"  {rankings_path}")

        # Log artifacts if requested
        if log_artifacts:
            if verbose:
                print(f"\nLogging insights to MLflow...")

            # Create run for insights
            with ExperimentTracker(
                "insights",
                auto_log_environment=False
            ) as tracker:
                insights_run_id = tracker.start_run(
                    f"insights_{run_ids_list[0][:8]}",
                    tags={"run_ids": ",".join(run_ids_list)}
                )

                # Log files as artifacts
                tracker.log_artifacts(str(output_path))

                # Log key metrics as params
                if "best_run" in insights:
                    tracker.log_param("best_run", insights["best_run"])

                if verbose:
                    print(f"Insights logged to run: {insights_run_id}")

        return 0

    except Exception as e:
        print(f"Insights generation error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main_insights(argv: Optional[list] = None) -> int:
    """CLI entry point for exp-insights command.

    Usage:
        exp-insights <run_ids>                                    # Generate insights
        exp-insights <run_ids> --metric val.mae                   # Use different metric
        exp-insights <run_ids> --group-by params.learning_rate    # Group by parameter
        exp-insights <run_ids> --min-sample-size 10              # Require more samples
        exp-insights <run_ids> --verbose                         # Detailed output
    """
    parser = argparse.ArgumentParser(
        description="Generate automated insights from experiment comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exp-insights abc123,def456,ghi789                 Analyze multiple runs
  exp-insights abc123,def456 --metric val.mae       Use different metric
  exp-insights abc123,def456 --group-by params.lr   Group by learning rate
  exp-insights abc123,def456 --min-sample-size 10   Require 10 samples
  exp-insights abc123,def456 -v                     Verbose output

For more information, see: examples/configs/analytics/README.md
        """
    )

    parser.add_argument(
        "run_ids",
        help="Comma-separated list of MLflow run IDs to analyze"
    )

    parser.add_argument(
        "--metric",
        default="val.rmse",
        help="Metric to analyze (default: val.rmse)"
    )

    parser.add_argument(
        "--group-by",
        default=None,
        help="Parameter to group by (e.g., params.learning_rate)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="insights",
        help="Output directory for reports (default: insights)"
    )

    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=5,
        help="Minimum runs for statistical testing (default: 5)"
    )

    parser.add_argument(
        "--no-log-artifacts",
        action="store_false",
        dest="log_artifacts",
        help="Don't log insights as MLflow artifacts"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed analysis information"
    )

    args = parser.parse_args(argv)

    return exp_insights_command(
        run_ids=args.run_ids,
        metric=args.metric,
        group_by=args.group_by,
        output_dir=args.output_dir,
        min_sample_size=args.min_sample_size,
        log_artifacts=args.log_artifacts,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
