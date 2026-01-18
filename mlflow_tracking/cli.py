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


if __name__ == "__main__":
    sys.exit(main())
