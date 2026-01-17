# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-17)

**Core value:** Understand what drives biomass predictions through systematic experimentation
**Current focus:** Phase 2: Experiment Organization & Discovery

## Current Position

Phase: 2 of 7 (Organization & Discovery)
Plan: 1 of 2 in current phase
Status: Phase 1 complete, Phase 2 in progress
Last activity: 2026-01-17 — Completed 02-01-PLAN.md (Organization & Discovery)

Progress: Phase 1: [██████████] 100% | Phase 2: [█████░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 4.2 min
- Total execution time: 0.4 hours

**By Phase:**

| Phase | Plans | Complete | Avg/Plan |
|-------|-------|----------|----------|
| 01-experiment-tracking-foundation | 4 | 4 | 4.0 min |
| 02-organization-discovery | 1 | 1 | 5.0 min |

**Recent Trend:**
- Last 5 plans: 4.2 min avg (5 plans)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**From 01-01-PLAN.md (MLflow Tracking Infrastructure):**
- Use absolute paths for MLflow tracking URI to ensure consistent database location
- Set global MLflow tracking URI in ExperimentTracker.__init__ for proper session management
- Implement context manager support for automatic status tracking on exceptions
- SQLite backend for local development (sufficient for single-user workflows)

**From 01-02-PLAN.md (Canonical Data Splits):**
- Image-level splitting (357 images) not target-level (1785 rows) to prevent data leakage
- Stratification using 5 quantile bins on Dry_Total_g target for distribution balance
- JSON persistence for canonical splits to enable experiment reproducibility

**From 01-03-PLAN.md (Environment & Reproducibility Tracking):**
- Log git commit hash and branch as MLflow tags (not params) since they're metadata
- Log package versions as params prefixed with 'env.' to distinguish from hyperparameters
- Default auto_log_environment=True to ensure all runs capture reproducibility metadata
- Handle missing git gracefully by returning 'unknown' instead of raising exception
- Random seed logged as separate param for easy access and filtering

**From 01-04-PLAN.md (Documentation and Examples):**
- Export main classes via __init__.py for clean user imports
- Provide synthetic data fallback in full_example.py for immediate usability
- Use metric prefixes (train.rmse, val.rmse, test.rmse) for clear split separation
- Document context manager as primary/recommended usage pattern
- Include BAD vs GOOD code examples in best practices to prevent common pitfalls

**From 02-01-PLAN.md (Organization and Discovery):**
- Use MLflow's built-in experiment/run model instead of custom storage
- Leverage MLflow search_runs() with filter strings for powerful querying
- Tag-based organization using MLflow tags (model_type, purpose, phase)
- Group-based experiment isolation using MLflow experiments
- Simplified dict return format for search results (not MLflow objects)
- Extend ExperimentTracker with convenience methods rather than separate utilities

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

None yet.

## Session Continuity

Last session: 2026-01-17 16:31 UTC
Completed: Phase 2 Plan 1 (Organization & Discovery) - ExperimentOrganizer class with grouping, tagging, and search
Next phase: Phase 2 Plan 2 (pending)
Resume file: None

## Phase 2 Deliverables

**In Progress: 2026-01-17**

### Organization (ORG-01 through ORG-04)
- ✓ ORG-01: Group experiments via ExperimentOrganizer.create_group()
- ✓ ORG-02: Tag experiments via add_tags() and add_tags_to_run()
- ✓ ORG-03: Search experiments via search_runs() with MLflow filter syntax
- ✓ ORG-04: Web UI via built-in MLflow UI at http://localhost:5000

### Files Created (Phase 2)
- `mlflow_tracking/organizer.py` - ExperimentOrganizer class (327 lines)
- `mlflow_tracking/test_organization.py` - Organization features demo (267 lines)
- `mlflow_tracking/README.md` - Updated with organization documentation (814 lines total)

### Files Modified (Phase 2)
- `mlflow_tracking/tracker.py` - Added add_tags(), set_group(), get_run_id() methods
- `mlflow_tracking/__init__.py` - Added ExperimentOrganizer and create_group exports

**Total Phase 2**: 1,108+ new lines of code and documentation

### Ready for Phase 2 Plan 2
Organization infrastructure complete. Next plan will add advanced comparison and analysis features.
