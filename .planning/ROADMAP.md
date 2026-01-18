### Phase 8: Advanced Analytics

**Goal**: Enable error analysis, model interpretability, and automated insights generation

**Depends on**: Phase 7

**Requirements**: ANALYTICS-01, ANALYTICS-02, ANALYTICS-03

**Success Criteria** (what must be TRUE):
1. User can analyze prediction errors and identify failure modes
2. User can interpret model decisions and feature importance
3. User can generate automated insights from experiment results

**Plans**: 4 plans in 4 waves

- [x] 08-01-PLAN.md — Create ErrorAnalyzer class for residual analysis and failure mode identification (Wave 1) ✓ Complete
- [x] 08-02-PLAN.md — Create ModelInterpretability class for SHAP and ELI5 explanations (Wave 2) ✓ Complete
- [x] 08-03-PLAN.md — Create InsightsGenerator class for automated insights and statistical testing (Wave 3) ✓ Complete
- [x] 08-04-PLAN.md — Create CLI commands and ReportGenerator for analytics workflows (Wave 4) ✓ Complete

**Details**:
Phase 8 implements advanced analytics capabilities building on the solid foundation of Phases 1-7. The plan follows a vertical slice approach: error analysis (08-01), model interpretability (08-02), automated insights (08-03), and CLI/reporting integration (08-04). Each plan is self-contained with 2-3 tasks, targeting ~50% context usage per plan.

**Wave Structure:**
- Wave 1: ErrorAnalyzer (residual plots, error distributions, failure mode clustering)
- Wave 2: ModelInterpretability (SHAP values, feature importance, permutation importance)
- Wave 3: InsightsGenerator (statistical testing, hyperparameter correlations, recommendations)
- Wave 4: CLI commands and ReportGenerator (exp-analyze-errors, exp-interpret, exp-insights, HTML/PDF reports)

**Integration Points:**
- Leverages MLflow artifacts (predictions CSVs, models) as single source of truth
- Extends ExperimentComparator pattern for data fetching
- Follows ErrorAnalyzer/ModelInterpretability/InsightsGenerator patterns
- CLI integration with existing exp-run, exp-run-batch, exp-run-optimize commands

---

### Phase 09: Analytics Data Pipeline Integration

**Goal**: Enable end-to-end analytics workflow by logging predictions artifacts

**Depends on**: Phase 8

**Requirements**: Closes Gap 1 from v1-MILESTONE-AUDIT.md

**Gap Closure**: Training → Artifacts → Analytics flow

**Success Criteria** (what must be TRUE):
1. Training scripts save predictions.csv to output directory
2. Adapters log predictions.csv as MLflow artifact after subprocess execution
3. ErrorAnalyzer.load_run() can retrieve predictions from MLflow artifacts
4. exp-analyze-errors CLI computes residuals from logged predictions

**Plans**: 1 plan in 1 wave

- [x] 09-01-PLAN.md — Add predictions artifact logging to adapters (Wave 1) ✓ Complete

**Details**:
Phase 9 closes the critical gap between training and analytics. Currently, training scripts execute via subprocess but don't save predictions artifacts, preventing Phase 8 analytics features from functioning. This phase adds predictions logging to both PyTorchAdapter and SklearnAdapter, enabling the complete analytics workflow.

**Wave Structure:**
- Wave 1: Add predictions_path to ExperimentConfig, log predictions artifact in adapters, test E2E flow

**Integration Points:**
- Modifies mlflow_tracking/adapters.py (PyTorchAdapter, SklearnAdapter)
- Modifies mlflow_tracking/config_parser.py (ExperimentConfig schema)
- Uses ErrorAnalyzer for verification of predictions loading
- Enables exp-analyze-errors CLI to function end-to-end

---

### Phase 10: Canonical Splits Enforcement

**Goal**: Enforce canonical data splits across all experiments for reproducibility

**Depends on**: Phase 4

**Requirements**: Closes REPRO-02 partial satisfaction, Gap 2 from v1-MILESTONE-AUDIT.md

**Gap Closure**: Canonical Splits → Training flow

**Success Criteria** (what must be TRUE):
1. Adapters load canonical splits from DataSplitter
2. Adapters pass split indices to training scripts via CLI args
3. Training scripts accept --train-indices, --val-indices, --test-indices flags
4. All experiments use identical train/validation/test splits

**Plans**: 1 plan in 1 wave

- [x] 10-01-PLAN.md — Integrate DataSplitter with adapters (Wave 1) ✓ Complete

**Details**:
Phase 10 closes the reproducibility gap by enforcing canonical splits. DataSplitter and canonical_splits.json exist from Phase 1, but adapters don't use them. This phase integrates DataSplitter with PyTorchAdapter and SklearnAdapter, ensuring all experiments use identical splits for fair comparison and reproducibility.

**Wave Structure:**
- Wave 1: Add split loading in adapters, pass via CLI args, modify training scripts to accept and use splits

**Integration Points:**
- Modifies mlflow_tracking/adapters.py (PyTorchAdapter, SklearnAdapter)
- Uses mlflow_tracking.data_split.DataSplitter (already exists)
- Modifies training scripts to accept --train-indices, --val-indices, --test-indices flags
- Passes indices as comma-separated strings via subprocess args

---

### Phase 11: Batch Organization Improvements

**Goal**: Organize batch runs into experiment groups for better discoverability

**Depends on**: Phase 6

**Requirements**: Closes Gap 3 from v1-MILESTONE-AUDIT.md

**Gap Closure**: ExperimentOrganizer → BatchExecutor integration

**Success Criteria** (what must be TRUE):
1. BatchExecutor creates experiment groups for batch runs
2. Group names are timestamp-based and descriptive
3. Batch metadata tags are added to groups
4. MLflow UI shows organized batch experiments

**Plans**: 1 plan in 1 wave

- [x] 11-01-PLAN.md — Integrate ExperimentOrganizer with BatchExecutor (Wave 1) ✓ Complete

**Details**:
Phase 11 improves organization by integrating ExperimentOrganizer with BatchExecutor. Currently, batch runs don't create experiment groups, making it difficult to find related runs. This phase adds group creation to batch execution, improving discoverability in MLflow UI.

**Wave Structure:**
- Wave 1: Add organizer.create_group() call in BatchExecutor, generate timestamp-based names, add metadata tags

**Integration Points:**
- Modifies mlflow_tracking/batch_executor.py (BatchExecutor)
- Uses mlflow_tracking.organizer.ExperimentOrganizer (already exists)
- Integrates with existing exp-run-batch CLI command

---

### Phase 12: Flexible Script Paths

**Goal**: Make training script paths configurable via YAML

**Depends on**: Phase 4

**Requirements**: Closes Gap 4 from v1-MILESTONE-AUDIT.md

**Gap Closure**: Tech debt - hardcoded script paths in adapters

**Success Criteria** (what must be TRUE):
1. ExperimentConfig includes script_path field
2. Adapters read script_path from config instead of hardcoding
3. Script path validation occurs before execution
4. Example configs demonstrate script_path usage

**Plans**: 1 plan in 1 wave

- [x] 12-01-PLAN.md — Add script_path to ExperimentConfig schema (Wave 1) ✓ Complete

**Details**:
Phase 12 removes hardcoded script paths from adapters, improving flexibility. Currently, PyTorchAdapter and SklearnAdapter have hardcoded paths like "scripts/train_oof_effnet.py". This phase adds script_path to the config schema, making it easy to run different scripts without adapter modifications.

**Wave Structure:**
- Wave 1: Add script_path to ExperimentConfig, update adapters to use config value, add validation, update examples

**Integration Points:**
- Modifies mlflow_tracking/config_parser.py (ExperimentConfig schema)
- Modifies mlflow_tracking/adapters.py (PyTorchAdapter, SklearnAdapter)
- Updates example configs to include script_path field
- Adds path existence validation in ConfigParser
