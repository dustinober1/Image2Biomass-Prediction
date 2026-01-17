"""
Command-line interface for running experiments from YAML configuration.

This module provides the `exp-run` command that loads YAML configurations,
validates them against registered adapters, and executes experiments using
MLflow for logging.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from mlflow_tracking import (
    ConfigParser,
    ExperimentTracker,
    AdapterRegistry
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


if __name__ == "__main__":
    sys.exit(main())
