# Roadmap: Image2Biomass Experimental Framework

## Overview

Transform 29 ad-hoc experiment scripts into a systematic experimentation framework that enables controlled ablations, model comparisons, and insights generation. The journey builds core tracking infrastructure first, adds organization and analysis capabilities, then creates a configuration-driven execution layer, and finally scales to parallel execution and hyperparameter optimization. Each phase delivers a complete, verifiable capability that builds toward the goal of understanding what drives biomass predictions through systematic experimentation rather than trial-and-error.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Experiment Tracking Foundation** - Core tracking and reproducibility infrastructure
- [ ] **Phase 2: Organization & Discovery** - Grouping, tagging, and searching experiments
- [ ] **Phase 3: Analysis & Comparison** - Metrics comparison and results aggregation
- [ ] **Phase 4: Configuration System** - YAML-driven experiment definitions
- [ ] **Phase 5: Script Adapters & Auto-Logging** - Integration with existing 29 training scripts
- [ ] **Phase 6: Parallel Execution Infrastructure** - Batch execution and resource management
- [ ] **Phase 7: Hyperparameter Optimization** - Optuna integration with pruning

## Phase Details

### Phase 1: Experiment Tracking Foundation

**Goal**: Establish reproducible experiment tracking with comprehensive logging and metadata capture

**Depends on**: Nothing (first phase)

**Requirements**: TRACK-01, TRACK-02, TRACK-03, TRACK-04, TRACK-05, REPRO-01, REPRO-02, REPRO-03

**Success Criteria** (what must be TRUE):
1. User can record every experiment execution with timestamp, status (running/completed/failed), and duration
2. User can capture all hyperparameters and configuration values for each experiment
3. User can record evaluation metrics (RMSE, R², MAE) per target variable
4. User can store artifacts (model files, predictions CSVs, analysis outputs) with experiment references
5. User can log experiments programmatically via Python SDK within training scripts
6. Framework tracks Python environment (package versions) for each experiment
7. Framework enforces proper data splitting (train/validation/test) to prevent leakage
8. Framework logs ALL experiments including failures (prevents cherry-picking)

**Plans**: 4 plans in 3 waves

- [ ] 01-01-PLAN.md — Initialize MLflow tracking infrastructure with Python SDK (Wave 1)
- [ ] 01-02-PLAN.md — Create canonical three-way data split utilities (Wave 1)
- [ ] 01-03-PLAN.md — Implement automatic environment and reproducibility tracking (Wave 2)
- [ ] 01-04-PLAN.md — Create comprehensive example and documentation (Wave 3, checkpoint:human-verify)

### Phase 2: Organization & Discovery

**Goal**: Enable systematic organization and discovery of experiments through groups, tags, and search

**Depends on**: Phase 1

**Requirements**: ORG-01, ORG-02, ORG-03, ORG-04

**Success Criteria** (what must be TRUE):
1. User can group related experiments (e.g., "ablation-studies", "ensemble-tests")
2. User can tag experiments by model type, phase, and purpose
3. User can search and filter experiments by metrics, parameters, and tags
4. User can view experiments, metrics, and artifacts via web UI

**Plans**: TBD

### Phase 3: Analysis & Comparison

**Goal**: Enable side-by-side comparison and aggregation of experimental results

**Depends on**: Phase 2

**Requirements**: ANALYSIS-01, ANALYSIS-02, ANALYSIS-03

**Success Criteria** (what must be TRUE):
1. User can compare metrics side-by-side across multiple experiments
2. User can aggregate results from multiple experiments into structured format
3. User can generate insights by clustering experiment results and identifying patterns

**Plans**: TBD

### Phase 4: Configuration System

**Goal**: Enable experiment definition via YAML configurations instead of code

**Depends on**: Phase 3

**Requirements**: CONFIG-01, CONFIG-02, CONFIG-03, INTEGRATION-01

**Success Criteria** (what must be TRUE):
1. User can define experiments as YAML configuration files
2. User can execute experiments via CLI: `exp-run config.yaml`
3. User can use parameter templating and variable substitution for systematic sweeps
4. Framework wraps existing 29 training scripts via adapter pattern (no script modifications required)

**Plans**: TBD

### Phase 5: Script Adapters & Auto-Logging

**Goal**: Automatic metric logging for common ML frameworks without manual code

**Depends on**: Phase 4

**Requirements**: INTEGRATION-02, INFRA-03

**Success Criteria** (what must be TRUE):
1. Framework auto-logs metrics for sklearn/XGBoost/PyTorch models without manual logging code
2. Framework manages random seeds for reproducibility (configurable per experiment)

**Plans**: TBD

### Phase 6: Parallel Execution Infrastructure

**Goal**: Enable batch execution of multiple experiments with resource management

**Depends on**: Phase 5

**Requirements**: INFRA-01, INFRA-02

**Success Criteria** (what must be TRUE):
1. Framework executes multiple experiments in parallel (batch mode)
2. Framework manages GPU/CPU resource allocation for concurrent experiments

**Plans**: TBD

### Phase 7: Hyperparameter Optimization

**Goal**: Integrate hyperparameter optimization with efficient search and pruning

**Depends on**: Phase 6

**Requirements**: OPT-01, OPT-02, OPT-03

**Success Criteria** (what must be TRUE):
1. Framework integrates hyperparameter optimization (Optuna) with efficient search
2. Framework supports pruning underperforming trials during optimization
3. Framework enables parallel execution of optimization trials

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Experiment Tracking Foundation | 0/4 | Not started | - |
| 2. Organization & Discovery | 0/TBD | Not started | - |
| 3. Analysis & Comparison | 0/TBD | Not started | - |
| 4. Configuration System | 0/TBD | Not started | - |
| 5. Script Adapters & Auto-Logging | 0/TBD | Not started | - |
| 6. Parallel Execution Infrastructure | 0/TBD | Not started | - |
| 7. Hyperparameter Optimization | 0/TBD | Not started | - |
