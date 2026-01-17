# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-17)

**Core value:** Understand what drives biomass predictions through systematic experimentation
**Current focus:** Phase 4: Configuration System (next phase)

## Current Position

Phase: 4 of 7 (Configuration System)
Plan: 2 of 4
Status: In progress
Last activity: 2026-01-17 — Completed Phase 4 Plan 2 (YAML Config Loader)

Progress: Phase 1: [██████████] 100% | Phase 2: [██████████] 100% | Phase 3: [██████████] 100% | Phase 4: [███░░░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 3.5 min
- Total execution time: 0.5 hours

**By Phase:**

| Phase | Plans | Complete | Avg/Plan |
|-------|-------|----------|----------|
| 01-experiment-tracking-foundation | 4 | 4 | 4.0 min |
| 02-organization-discovery | 1 | 1 | 5.0 min |
| 03-analysis-comparison | 1 | 1 | 3.0 min |
| 04-configuration-system | 2 | 2 | 2.9 min |

**Recent Trend:**
- Last 5 plans: 3.5 min avg (8 plans total)

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

**From 03-01-PLAN.md (Analysis and Comparison):**
- Use scikit-learn KMeans for clustering (standard, well-documented)
- Use scipy.stats.zscore with fallback to manual implementation for compatibility
- Auto-sort comparison results by primary metric (val.rmse) for quick analysis
- Return both DataFrame and dict formats via as_dataframe parameter for flexibility
- All comparison methods return consistent structure (DataFrame or dict)
- Export methods are standalone (work on any DataFrame, not just comparison results)
- NaN handling via column mean filling for clustering, dropping for correlation

**From 04-01-PLAN.md (YAML Schema and Adapter Interface):**
- Use Pydantic for schema validation (automatic validation, type coercion, clear error messages)
- Quote Jinja2 template variables in YAML to avoid parsing errors (e.g., '{{lr}}')
- Implement strict validation with extra='forbid' to prevent typos in config files
- Separate schema definition from YAML loading for clear separation of concerns
- Adapter pattern with @AdapterRegistry.register() decorator for training script wrapping

**From 04-02-PLAN.md (YAML Config Loader):**
- Use yaml.safe_load() for security (prevents code execution from malicious YAML)
- Use Jinja2 Template for variable substitution ({{var}} syntax, widely-used)
- Use itertools.product() for grid search expansion (all combinations efficiently)
- Remove sweep section from individual expanded configs (cleaner output)
- Static methods for ConfigParser (no instance state needed, simpler API)

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

None yet.

## Session Continuity

Last session: 2026-01-17 17:20 UTC
Completed: Phase 4 Plan 2 (YAML Config Loader) - All 3 tasks complete
Next: Phase 4 Plan 3 (CLI Tool)
Resume file: None

## Phase 3 Deliverables

**Completed: 2026-01-17**

All 3 Phase 3 requirements satisfied:

### Analysis (ANALYSIS-01 through ANALYSIS-03)
- ✓ ANALYSIS-01: Compare metrics side-by-side across multiple experiments via compare_by_ids, compare_by_group, compare_by_filter
- ✓ ANALYSIS-02: Aggregate results from multiple experiments into structured format (DataFrame/dict) with export to CSV/JSON/Excel
- ✓ ANALYSIS-03: Generate insights by clustering experiment results (K-means), identifying patterns (correlation), detecting outliers

### Files Created
- `mlflow_tracking/comparison.py` - ExperimentComparator class (732 lines)
- `mlflow_tracking/test_comparison.py` - Comparison & analysis features demo (420 lines)

### Files Modified
- `mlflow_tracking/__init__.py` - Added ExperimentComparator export
- `mlflow_tracking/README.md` - Added Comparison & Analysis section, updated requirements coverage (+182 lines)

**Total Phase 3**: 1,334+ lines of code and documentation

### Ready for Next Phase
Analysis and comparison infrastructure complete. Users can now compare experiments side-by-side, aggregate results, and generate insights through clustering, correlation, and outlier detection.

## Phase 4 Deliverables

**Started: 2026-01-17**

### Plan 1: YAML Schema and Adapter Interface (04-01)

**Completed: 2026-01-17**

All requirements satisfied:

- ✓ ExperimentConfig schema defined with Pydantic validation
- ✓ BaseAdapter abstract interface created with execute() protocol
- ✓ AdapterRegistry provides registration and retrieval
- ✓ Example YAML configurations demonstrate schema usage
- ✓ All classes exported from mlflow_tracking package
- ✓ Tests confirm schema validation catches invalid configs

### Files Created
- `mlflow_tracking/config_parser.py` - ExperimentConfig schema with validation (221 lines)
- `mlflow_tracking/adapters.py` - BaseAdapter and AdapterRegistry (229 lines)
- `examples/configs/basic_experiment.yaml` - Basic example config
- `examples/configs/sweep_experiment.yaml` - Parameter sweep example
- `examples/configs/README.md` - Schema documentation and usage guide (147 lines)

### Files Modified
- `mlflow_tracking/__init__.py` - Added exports for ExperimentConfig, BaseAdapter, AdapterRegistry

**Total Plan 01**: 597+ lines of code and documentation

### Ready for Next Plan
Schema and adapter interface complete. Ready to implement YAML config loader (04-02) that will parse YAML files and create ExperimentConfig instances.

### Plan 2: YAML Config Loader (04-02)

**Completed: 2026-01-17**

All requirements satisfied:

- ✓ ConfigParser class with load_config(), expand_sweeps(), validate() methods
- ✓ YAML configuration loading with Pydantic schema validation
- ✓ Parameter sweep expansion using Jinja2 templating and itertools.product
- ✓ Adapter validation integration with AdapterRegistry
- ✓ Comprehensive test script demonstrating all features
- ✓ ConfigParser exported from mlflow_tracking package

### Files Created
- `mlflow_tracking/config_parser.py` - ConfigParser class with loading, expansion, validation (397 lines)
- `mlflow_tracking/test_config_parser.py` - Comprehensive test script (194 lines)

### Files Modified
- `mlflow_tracking/__init__.py` - Added ConfigParser export

**Total Plan 02**: 591+ lines of code

### Ready for Next Plan
ConfigParser complete with YAML loading, sweep expansion, and adapter validation. Ready to implement CLI tool (04-03) that will provide command-line interface for running experiments from config files.
