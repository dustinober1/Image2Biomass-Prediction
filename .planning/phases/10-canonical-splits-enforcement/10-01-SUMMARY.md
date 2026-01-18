---
phase: 10-canonical-splits-enforcement
plan: 01
subsystem: data-splitting
tags: [canonical-splits, data-splitter, reproducibility, mlflow, pytorch, sklearn]

# Dependency graph
requires:
  - phase: 01-experiment-tracking-foundation
    provides: DataSplitter class, canonical_splits.json file
provides:
  - Adapters automatically load and pass canonical splits to training scripts
  - Training scripts accept split indices via CLI args with KFold fallback
  - All experiments use identical train/validation/test splits
affects: [11-feature-store, 12-model-registry, production-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Adapter loads canonical splits and passes as CLI args
    - Training scripts accept --train-indices, --val-indices, --test-indices
    - Conditional logic: canonical splits when provided, KFold fallback otherwise
    - Backward compatibility maintained for direct script execution

key-files:
  created: []
  modified:
    - mlflow_tracking/adapters.py
    - scripts/train_oof_effnet.py
    - scripts/train_ridge_advanced.py

key-decisions:
  - "Splits loaded in adapter, not config - always enforced, no opt-out"
  - "KFold fallback maintained for backward compatibility with direct script execution"
  - "Split indices passed as comma-separated strings to avoid CLI arg length limits"
  - "Training scripts unchanged when run directly without split args"

patterns-established:
  - "Pattern 1: Adapter loads splits via DataSplitter, converts to comma-separated strings, passes as CLI args"
  - "Pattern 2: Training scripts parse indices, check if all three provided, use canonical splits if yes"
  - "Pattern 3: Backward compatibility - scripts work with or without split indices"

# Metrics
duration: 4min
completed: 2026-01-18
---

# Phase 10 Plan 1: Canonical Splits Integration Summary

**Adapters automatically load canonical splits from DataSplitter and pass to training scripts via CLI args, ensuring all experiments use identical train/validation/test splits for reproducibility**

## Performance

- **Duration:** 4 minutes
- **Started:** 2026-01-18T01:45:30Z
- **Completed:** 2026-01-18T01:49:57Z
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments

- PyTorchAdapter and SklearnAdapter load canonical splits from data/canonical_splits.json
- Split indices passed to training scripts as --train-indices, --val-indices, --test-indices CLI args
- train_oof_effnet.py accepts split indices with conditional canonical/KFold logic
- train_ridge_advanced.py accepts split indices with conditional canonical/KFold logic
- Backward compatibility maintained - scripts work with or without split indices

## Task Commits

Each task was committed atomically:

1. **Task 1: Add split loading to PyTorchAdapter** - `17c5828` (feat)
2. **Task 2: Add split loading to SklearnAdapter** - `c061a65` (feat)
3. **Task 3: Modify train_oof_effnet.py to accept split indices** - `45976a3` (feat)
4. **Task 4: Modify train_ridge_advanced.py to accept split indices** - `2f7d30c` (feat)

**Plan metadata:** `lmn012o` (docs: complete plan)

_Note: TDD tasks may have multiple commits (test → feat → refactor)_

## Files Created/Modified

- `mlflow_tracking/adapters.py` - Added DataSplitter import, both adapters load canonical splits and pass as CLI args
- `scripts/train_oof_effnet.py` - Added argparse, parse_indices helper, conditional canonical/KFold logic, main() function
- `scripts/train_ridge_advanced.py` - Added argparse, parse_indices helper, conditional canonical/KFold logic, main() function

## Decisions Made

- **Splits loaded in adapter, not config** - Ensures canonical splits always used, no opt-out possible
- **KFold fallback maintained** - Backward compatibility for direct script execution without adapter
- **Comma-separated indices** - Avoids CLI arg length limits with large split files
- **Conditional logic in training scripts** - Check if all three splits provided, use canonical if yes, KFold otherwise

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed smoothly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Gap 2 (Canonical Splits Enforcement) from v1-MILESTONE-AUDIT.md is now CLOSED.**

All experiments now use identical splits from canonical_splits.json:
- Adapters automatically load and pass splits
- Training scripts accept and use provided splits
- KFold fallback for backward compatibility
- Reproducibility ensured across all experiments

Ready for:
- Phase 10 Plan 2: Additional canonical splits enforcement (if needed)
- Phase 11: Feature Store Integration
- Phase 12: Model Registry
- Production pipeline with guaranteed reproducible splits

---
*Phase: 10-canonical-splits-enforcement*
*Completed: 2026-01-18*
