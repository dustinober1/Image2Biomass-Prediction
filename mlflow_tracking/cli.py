"""
Command-line interface for running experiments from YAML configuration.

This module provides the `exp-run` command that loads YAML configurations,
validates them against registered adapters, and executes experiments using
MLflow for logging.

Also provides `exp-run-batch` command for parallel batch execution.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

from mlflow_tracking import (
    ConfigParser,
    ExperimentTracker,
    AdapterRegistry,
    BatchExecutor,
    ResourceManager
)


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


if __name__ == "__main__":
    sys.exit(main())
