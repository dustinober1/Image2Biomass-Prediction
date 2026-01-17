# Phase 3: Analysis & Comparison - Context

**Gathered:** 2026-01-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable side-by-side comparison and aggregation of experimental results. Users can compare multiple experiments, aggregate metrics, and generate insights through clustering, correlation, and outlier detection. This phase focuses on query and analysis of existing experiments, not on running new experiments or storing new data.

</domain>

<decisions>
## Implementation Decisions

### Comparison interface
- Primary method: By run IDs — user provides explicit run IDs for comparison
- Separate functions for each method: `compare_by_ids()`, `compare_by_group()`, `compare_by_filter()`
- Returns all rows from all runs (columns may vary between runs)
- Include all columns by default: params, metrics, tags, metadata

### Output format
- Both DataFrame and dict return types available via parameter (e.g., `as_dataframe=True`)
- Dict structure uses wide format: rows=experiments, columns=metrics/params
- Separate `to_csv()`, `to_json()`, `to_excel()` methods on result object for export
- Auto-sort by primary metric (e.g., `val.rmse` ascending)

### Metrics handling
- Return all data for each run with varying columns (let user handle mismatches)
- Raise error if key metrics are missing from some runs (configurable list)
- Failed runs are included with status column and NaN/None metrics
- Required metrics: Claude's discretion to determine reasonable defaults based on project

### Insights generation
- Three insight types: K-means clustering, correlation analysis, outlier identification
- Separate functions for each insight type: `cluster_runs()`, `correlate_params()`, `find_outliers()`
- Analyze all logged metrics by default (comprehensive)

### Claude's Discretion
- Default required metrics list for validation (based on Phase 1 canonical split expectations)
- Number of clusters for K-means (elbow method or simple heuristic)
- Correlation threshold for highlighting significant relationships
- Outlier detection method (z-score, IQR, or isolation forest)
- Exact NaN handling in DataFrames vs dict representations

</decisions>

<specifics>
## Specific Ideas

- Need to support comparing the 29 existing experiments already logged in MLflow
- Primary targets are `Dry_Total_g` and `Fresh_Total_g` biomass predictions
- Users will want to quickly identify which hyperparameter combinations perform best
- Analysis should help answer "what drives biomass predictions" through pattern identification

</specifics>

<deferred>
## Deferred Ideas

- Visualization/plots of comparison results — separate phase or follow-up work
- Statistical significance testing between experiment groups — future enhancement
- Time-series analysis of experiment performance over time — future enhancement

</deferred>

---

*Phase: 03-analysis-comparison*
*Context gathered: 2026-01-17*
