---
phase: 04-configuration-system
plan: 03
subsystem: cli
tags: [argparse, setuptools, entry-points, yaml, mlflow]

# Dependency graph
requires:
  - phase: 04-01
    provides: ExperimentConfig schema, BaseAdapter interface, AdapterRegistry
  - phase: 04-02
    provides: ConfigParser with load_config, expand_sweeps, validate methods
  - phase: 01-01
    provides: ExperimentTracker with MLflow logging and mark_failed method
provides:
  - exp-run CLI command for executing experiments from YAML configs
  - setup.py with package installation and console_scripts entry point
  - test_cli.py demonstrating CLI and programmatic usage
  - Programmatic API via exp_run_command() function
affects: [04-04-concrete-adapters]

# Tech tracking
tech-stack:
  added: [argparse (stdlib), setuptools with console_scripts]
  patterns: [CLI entry point via setup.py, error handling with tracker.mark_failed(), argparse with --sweep and --verbose flags]

key-files:
  created: [mlflow_tracking/cli.py, setup.py, mlflow_tracking/test_cli.py]
  modified: [mlflow_tracking/__init__.py, mlflow_tracking/tracker.py]

key-decisions:
  - "Use setuptools console_scripts for CLI installation (standard Python packaging)"
  - "Add mark_failed() method to ExperimentTracker for explicit error tracking"
  - "Return exit codes (0=success, 1=failure) for shell scripting integration"
  - "Export CLI functions from package for programmatic usage"

patterns-established:
  - "Pattern: CLI entry point maps to package.module:function"
  - "Pattern: Adapter errors wrapped in try/except with tracker.mark_failed()"
  - "Pattern: Verbose flag for detailed execution output"

# Metrics
duration: 2min
completed: 2026-01-17
---

# Phase 4 Plan 3: CLI Tool Summary

**exp-run CLI command with argparse for executing experiments from YAML configs, setuptools entry point installation, and ExperimentTracker integration for automatic MLflow logging**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-17T17:22:09Z
- **Completed:** 2026-01-17T17:24:44Z
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments

- Created exp-run CLI command that loads and executes YAML configurations via ConfigParser
- Integrated ExperimentTracker for automatic MLflow logging with mark_failed() error handling
- Added setup.py with console_scripts entry point for pip installation
- Implemented programmatic API via exp_run_command() for Python code integration
- Created test script demonstrating CLI usage with clear adapter requirements

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement CLI entry point** - `66f82f6` (feat)
2. **Task 2: Create setup.py with CLI entry point** - `0bedcba` (feat)
3. **Task 3: Create test script** - `fbea3cb` (feat)
4. **Task 4: Update package exports** - `698a043` (feat)

**Plan metadata:** (pending final commit)

## Files Created/Modified

### Created
- `mlflow_tracking/cli.py` - CLI entry point with main() and exp_run_command()
- `setup.py` - Package configuration with console_scripts entry point
- `mlflow_tracking/test_cli.py` - Test script demonstrating CLI usage

### Modified
- `mlflow_tracking/__init__.py` - Added exports for main and exp_run_command
- `mlflow_tracking/tracker.py` - Added mark_failed() method for error tracking

## Decisions Made

- **Use argparse for CLI parsing** - Standard library, well-documented, supports help generation
- **Return exit codes (0/1)** - Enables shell scripting integration and CI/CD pipelines
- **Add mark_failed() to ExperimentTracker** - Provides explicit error tracking with error_message tag in MLflow UI
- **Export CLI functions from package** - Enables both CLI usage (`exp-run config.yaml`) and programmatic usage (`exp_run_command(...)`)
- **Use setuptools console_scripts** - Standard Python packaging approach for CLI installation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added mark_failed() method to ExperimentTracker**

- **Found during:** Task 1 (CLI implementation)
- **Issue:** CLI calls `tracker.mark_failed()` but method didn't exist in ExperimentTracker class
- **Fix:** Added mark_failed() method that:
  - Checks for active run (raises RuntimeError if none)
  - Logs error_message as MLflow tag if provided
  - Calls end_run(status="failed") to mark run as failed
- **Files modified:** `mlflow_tracking/tracker.py`
- **Verification:** Method signature validated: `(self, error_message: str | None = None) -> None`
- **Committed in:** `66f82f6` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Auto-fix necessary for correct operation. CLI requires explicit error marking for MLflow tracking. No scope creep.

## Issues Encountered

- **Environment management** - System Python is externally managed, required using existing virtual environment at `.venv/` for package installation
- **Missing dependencies during testing** - Initial import test failed due to mlflow not being available in system Python, resolved by using virtual environment

## User Setup Required

None - no external service configuration required.

**Installation required:**
```bash
# Install package in development mode
pip install -e .

# Or using virtual environment
source .venv/bin/activate
pip install -e .
```

## Next Phase Readiness

### Ready for Next Plan (04-04: Concrete Adapters)

- CLI infrastructure complete and tested
- AdapterRegistry has `pytorch` and `sklearn` adapters registered (from prior plans)
- exp-run command ready to execute experiments once adapters have full implementations
- Error handling pattern established: `tracker.mark_failed()` called on adapter exceptions

### Blockers/Concerns

- **None** - CLI tool is ready for adapter implementation in next plan

### Integration Points for Next Plan

- Next plan (04-04) will implement concrete adapter execute() methods
- CLI will call `adapter.execute(config, tracker)` and expect dict of metrics
- Adapters should raise exceptions on failure (CLI will catch and mark as failed)

---
*Phase: 04-configuration-system*
*Completed: 2026-01-17*
