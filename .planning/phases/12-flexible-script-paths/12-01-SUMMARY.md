---
phase: 12-flexible-script-paths
plan: 01
subsystem: config
tags: pydantic, yaml, path-validation, adapters

# Dependency graph
requires:
  - phase: 04-configuration-system
    provides: ExperimentConfig schema with Pydantic validation
provides:
  - Configurable script paths via YAML (removes hardcoded tech debt)
  - Path existence validation with .py extension check
  - Backward-compatible adapters with deprecation warnings
  - Example configs demonstrating script_path usage
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Config field with optional default and path existence validation
    - Adapter fallback pattern for backward compatibility
    - Deprecation warnings for optional config fields

key-files:
  created: []
  modified:
    - mlflow_tracking/config_parser.py
    - mlflow_tracking/adapters.py
    - examples/configs/adapter_examples/pytorch_effnet.yaml
    - examples/configs/adapter_examples/sklearn_ridge.yaml
    - examples/configs/README.md

key-decisions:
  - Made script_path optional (default None) for backward compatibility
  - Added path existence validation only when script_path is provided
  - Used adapter fallback pattern (config.script_path or "default_path")
  - Added deprecation warnings for configs without script_path

patterns-established:
  - "Optional field pattern: Field with default None + validator that returns early if None"
  - "Adapter fallback: config.field or 'hardcoded_default' for backward compatibility"
  - "Deprecation warnings: warnings.warn() in validate_config() for future requirements"

# Metrics
duration: 4min
completed: 2026-01-18
---

# Phase 12 Plan 1: Flexible Script Paths Summary

**Configurable script paths via YAML with path existence validation, eliminating hardcoded training script paths from adapters**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-18T12:32:32Z
- **Completed:** 2026-01-18T12:36:23Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added script_path field to ExperimentConfig schema with path existence and .py extension validation
- Updated PyTorchAdapter and SklearnAdapter to use config.script_path with backward-compatible fallbacks
- Updated example configs (pytorch_effnet.yaml, sklearn_ridge.yaml) to demonstrate script_path usage
- Documented script_path in examples/configs/README.md with best practices
- **Closed Gap 4 from v1-MILESTONE-AUDIT.md:** Removed hardcoded script paths (tech debt eliminated)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add script_path field to ExperimentConfig schema** - `3d3d095` (feat)
2. **Task 2: Update adapters to use config.script_path** - `79fee5f` (feat)
3. **Task 3: Update example configs with script_path** - `2dd53bd` (feat)

**Plan metadata:** (pending final docs commit)

## Files Created/Modified

- `mlflow_tracking/config_parser.py` - Added script_path field with Optional[str] type and validate_script_path field_validator
- `mlflow_tracking/adapters.py` - Updated PyTorchAdapter and SklearnAdapter to use config.script_path with fallback defaults, added deprecation warnings in validate_config()
- `examples/configs/adapter_examples/pytorch_effnet.yaml` - Added script_path: "scripts/train_oof_effnet.py"
- `examples/configs/adapter_examples/sklearn_ridge.yaml` - Added script_path: "scripts/train_ridge_advanced.py"
- `examples/configs/README.md` - Documented script_path in optional fields table, added best practice note, updated adapter examples

## Decisions Made

- **Made script_path optional:** Set default=None to maintain backward compatibility with existing configs
- **Path validation only when provided:** validate_script_path returns early if v is None, only validates file existence when script_path is specified
- **Adapter fallback pattern:** Used `config.script_path or "default_path"` to provide backward-compatible fallbacks while encouraging migration to script_path
- **Deprecation warnings:** Added warnings.warn() in validate_config() for both adapters to notify users that script_path will be required in future versions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for next phase:**

- Hardcoded script paths eliminated from adapters (Gap 4 closed)
- Backward compatibility maintained with fallback defaults and deprecation warnings
- Path validation prevents execution with non-existent scripts
- Example configs demonstrate recommended pattern
- Documentation updated with script_path best practices

**No blockers or concerns.**

The flexible script paths feature is complete and ready for use. Users can now specify training script paths in YAML configs instead of modifying adapter code, making it easy to run different scripts without code changes.

---
*Phase: 12-flexible-script-paths*
*Completed: 2026-01-18*
