# Requirements: Image2Biomass Experimental Framework

**Defined:** 2025-01-17
**Core Value:** Understand what drives biomass predictions through systematic experimentation

## v1 Requirements

Requirements for experimental framework. Each maps to roadmap phases.

### Core Tracking

- [ ] **TRACK-01**: Framework records each experiment execution with timestamp, status (running/completed/failed), and duration
- [ ] **TRACK-02**: Framework captures all hyperparameters and configuration values for each experiment
- [ ] **TRACK-03**: Framework records evaluation metrics (RMSE, R², MAE) per target variable
- [ ] **TRACK-04**: Framework stores artifacts (model files, predictions CSVs, analysis outputs) with experiment references
- [ ] **TRACK-05**: Framework provides Python SDK for programmatic logging within training scripts

### Organization & Discovery

- [ ] **ORG-01**: Framework supports grouping related experiments (e.g., "ablation-studies", "ensemble-tests")
- [ ] **ORG-02**: Framework supports tagging experiments by model type, phase, and purpose
- [ ] **ORG-03**: Framework enables searching/filtering experiments by metrics, parameters, and tags
- [ ] **ORG-04**: Framework provides web UI for viewing experiments, metrics, and artifacts

### Comparison & Analysis

- [ ] **ANALYSIS-01**: Framework enables side-by-side metric comparison across multiple experiments
- [ ] **ANALYSIS-02**: Framework aggregates results from multiple experiments into structured format
- [ ] **ANALYSIS-03**: Framework generates insights by clustering experiment results and identifying patterns

### Configuration-Driven

- [ ] **CONFIG-01**: Framework defines experiments as YAML configuration files (not code)
- [ ] **CONFIG-02**: Framework executes experiments via CLI: `exp-run config.yaml`
- [ ] **CONFIG-03**: Framework supports parameter templating and variable substitution for systematic sweeps

### Integration

- [ ] **INTEGRATION-01**: Framework wraps existing 29 training scripts via adapter pattern (no script modifications required)
- [ ] **INTEGRATION-02**: Framework auto-logs metrics for sklearn/XGBoost/PyTorch models without manual logging code

### Infrastructure

- [ ] **INFRA-01**: Framework executes multiple experiments in parallel (batch mode)
- [ ] **INFRA-02**: Framework manages GPU/CPU resource allocation for concurrent experiments
- [ ] **INFRA-03**: Framework manages random seeds for reproducibility (configurable per experiment)

### Reproducibility

- [ ] **REPRO-01**: Framework tracks Python environment (package versions) for each experiment
- [ ] **REPRO-02**: Framework enforces proper data splitting (train/validation/test) to prevent leakage
- [ ] **REPRO-03**: Framework logs ALL experiments including failures (prevents cherry-picking)

### Optimization

- [ ] **OPT-01**: Framework integrates hyperparameter optimization (Optuna) with efficient search
- [ ] **OPT-02**: Framework supports pruning underperforming trials during optimization
- [ ] **OPT-03**: Framework enables parallel execution of optimization trials

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analytics

- **ADVANCED-01**: Statistical significance testing for model comparisons (paired t-tests, bootstrap)
- **ADVANCED-02**: Confidence intervals for all metrics
- **ADVANCED-03**: Automatic report generation with visualizations

### Collaboration

- **COLLAB-01**: Cloud sync for experiment artifacts (optional W&B integration)
- **COLLAB-02**: Sharing experiment configurations and results via URLs
- **COLLAB-03**: Team access controls and permissions

### Deployment

- **DEPLOY-01**: Model serving endpoints for best-performing models
- **DEPLOY-02**: A/B testing framework for production models
- **DEPLOY-03**: Monitoring and alerting for model drift

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time WebSocket updates | Adds complexity; batch experimentation doesn't need live updates |
| Cloud-only storage | Local-first design; optional cloud sync in v2 |
| All-in-one MLOps platform | Focus is experimentation, not full ML lifecycle |
| Built-in compute provisioning | Assume user manages their own GPU/CPU resources |
| Custom model architectures | Framework wraps existing scripts; doesn't replace model development |
| Data collection/augmentation | Dataset is fixed (357 images); framework manages experiments only |
| Mobile/web app UI | Research tool; CLI + basic web UI sufficient |
| Automated feature engineering | Domain-specific; manual feature extraction stays in scripts |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRACK-01 | Phase 1 | Complete |
| TRACK-02 | Phase 1 | Complete |
| TRACK-03 | Phase 1 | Complete |
| TRACK-04 | Phase 1 | Complete |
| TRACK-05 | Phase 1 | Complete |
| ORG-01 | Phase 2 | Pending |
| ORG-02 | Phase 2 | Pending |
| ORG-03 | Phase 2 | Pending |
| ORG-04 | Phase 2 | Pending |
| ANALYSIS-01 | Phase 3 | Pending |
| ANALYSIS-02 | Phase 3 | Pending |
| ANALYSIS-03 | Phase 3 | Pending |
| CONFIG-01 | Phase 4 | Pending |
| CONFIG-02 | Phase 4 | Pending |
| CONFIG-03 | Phase 4 | Pending |
| INTEGRATION-01 | Phase 4 | Pending |
| INTEGRATION-02 | Phase 5 | Pending |
| INFRA-01 | Phase 6 | Pending |
| INFRA-02 | Phase 6 | Pending |
| INFRA-03 | Phase 5 | Pending |
| REPRO-01 | Phase 1 | Complete |
| REPRO-02 | Phase 1 | Complete |
| REPRO-03 | Phase 1 | Complete |
| OPT-01 | Phase 7 | Pending |
| OPT-02 | Phase 7 | Pending |
| OPT-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

---
*Requirements defined: 2025-01-17*
*Last updated: 2025-01-17 after Phase 1 completion*
