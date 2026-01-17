---
phase: 01-experiment-tracking-foundation
plan: 02
subsystem: data-management
tags: [data-splitting, stratification, scikit-learn, reproducibility, canonical-splits]

# Dependency graph
requires: []
provides:
  - Canonical train/validation/test split indices for 357 images
  - DataSplitter class for reproducible data splitting
  - Stratified splitting maintaining target distribution
affects: [experiment-framework, model-training, cross-validation]

# Tech tracking
tech-stack:
  added: [scikit-learn train_test_split, pandas qcut for stratification]
  patterns: [image-level splitting to prevent leakage, JSON persistence for canonical splits, stratified regression using quantile bins]

key-files:
  created: [mlflow_tracking/data_split.py, data/canonical_splits.json, mlflow_tracking/test_splits.py]
  modified: [mlflow_tracking/__init__.py]

key-decisions:
  - "Image-level splitting (357 images) not target-level (1785 rows) to prevent data leakage"
  - "Stratification using 5 quantile bins on Dry_Total_g target for distribution balance"
  - "JSON persistence for canonical splits to enable experiment reproducibility"

patterns-established:
  - "Pattern: DataSplitter class manages split lifecycle (create/validate/save/load)"
  - "Pattern: Stratified regression using pd.qcut for binning continuous targets"
  - "Pattern: Validation checks for overlap and distribution balance"

# Metrics
duration: 1min 23sec
completed: 2026-01-17
---

# Phase 1 Plan 2: Canonical Data Splits Summary

**Image-level stratified 70/15/15 train/validation/test splits with quantile-based stratification and JSON persistence**

## Performance

- **Duration:** 1 min 23 sec
- **Started:** 2026-01-17T15:53:37Z
- **Completed:** 2026-01-17T15:55:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- DataSplitter class with three-way splitting, validation, and persistence
- Canonical splits saved (249/54/54 images) with stratification
- Distribution balance validated (all splits within 2.6% of mean)
- Zero overlap between splits verified

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement DataSplitter class** - `2a487ed` (feat)
2. **Task 2: Create canonical splits and validation** - `9f66e41` (feat)

**Plan metadata:** (to be committed after SUMMARY.md)

## Files Created/Modified

- `mlflow_tracking/data_split.py` - DataSplitter class with create_splits, load_splits, validate_splits methods
- `data/canonical_splits.json` - Canonical train/val/test indices for 357 images
- `mlflow_tracking/test_splits.py` - Script to generate and validate canonical splits
- `mlflow_tracking/__init__.py` - Updated with flexible imports for modular loading

## Decisions Made

**Image-level splitting:** Chose to split at image level (357 images) rather than target level (1785 rows) to prevent data leakage. Each image has 5 target predictions, so splitting by rows would cause the same image to appear in multiple splits, leaking information between train/val/test sets.

**Stratification approach:** Used 5 quantile bins via pd.qcut on Dry_Total_g target for stratified splitting. This ensures similar biomass distribution across all three splits despite working with regression data (not classification).

**Validation criteria:** Splits validated for (1) zero overlap between sets, (2) proper 70/15/15 ratio, (3) distribution balance within 10% of overall mean. All criteria met with excellent balance (train 1.1%, val 2.6%, test 2.4% deviation).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed mlflow_tracking/__init__.py import error**

- **Found during:** Task 2 (running test_splits.py)
- **Issue:** __init__.py imported ExperimentTracker from mlflow_tracking.tracker which doesn't exist yet (from previous plan). This blocked importing data_split module.
- **Fix:** Updated __init__.py with flexible try/except imports that gracefully handle missing modules. Both ExperimentTracker (future) and DataSplitter (current) can coexist.
- **Files modified:** mlflow_tracking/__init__.py
- **Verification:** test_splits.py runs successfully, imports work correctly
- **Committed in:** 9f66e41 (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Import error blocked execution; flexible __init__.py enables incremental development. No scope creep.

## Issues Encountered

- **Import error in __init__.py:** File from previous plan referenced non-existent tracker module. Fixed with flexible imports enabling modules to be added incrementally.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready:**
- Canonical splits available for all experiments
- DataSplitter API documented and tested
- Import structure supports incremental module addition

**Usage for experiments:**
```python
from mlflow_tracking.data_split import DataSplitter
splitter = DataSplitter()
splitter.load_splits()
train_idx, val_idx, test_idx = splitter.get_split_indices()
```

**No blockers or concerns.** REPRO-02 requirement satisfied: framework enforces proper data splitting preventing leakage.

---
*Phase: 01-experiment-tracking-foundation*
*Completed: 2026-01-17*
