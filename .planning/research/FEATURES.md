# Feature Research: ML Experimentation Platforms

**Domain:** Machine Learning Experimentation Frameworks
**Researched:** 2026-01-17
**Confidence:** HIGH

**Source Analysis:**
- MLflow official documentation (HIGH confidence)
- Weights & Biases official documentation (HIGH confidence)
- Comet official documentation (HIGH confidence)
- Project analysis: 29 ad-hoc experiment scripts (GROUND TRUTH)

---

## Table Stakes (Users Expect These)

Features users assume exist. Missing these = platform feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Run Tracking** | Every experiment is a "run" - must record start/end time, status | Low | Core abstraction: execution with metadata |
| **Parameter Logging** | Hyperparameters define the experiment - must be queryable | Low | Flat key-value store, nested structures optional |
| **Metric Logging** | Results (loss, accuracy, RMSE) - must be visualizable over time | Low | Time-series data, real-time updates expected |
| **Artifact Storage** | Models, plots, predictions must be saved with the run | Medium | File storage with versioning, needs URI/path tracking |
| **Experiment Grouping** | Runs must be organized into logical groups (projects) | Low | Hierarchical: Project → Experiment → Run |
| **Tagging System** | Users need to filter/search runs by custom labels | Low | Arbitrary key-value tags for organization |
| **Web UI** | Visual exploration is essential for comparison | Medium | Table view + detail view + charts |
| **Search/Filter** | Finding runs by metric value or parameter is critical | Low | SQL-like queries: "rmse < 10 AND lr > 0.001" |
| **CLI Access** | Automation and scripting require command-line tools | Medium | Basic CRUD operations, listing, querying |
| **Python SDK** | Integration with training code requires Python API | Medium | Context managers (`with start_run()`) expected |

**Table Stakes Rationale:**
- Current project has 29 scattered scripts - each is an untracked "run"
- Pain point: "Results scattered across files" = no artifact storage
- Pain point: "Hard to identify what works" = no search/filter/grouping

---

## Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Configuration-Driven Experiments** | Define experiments as YAML/config, not scripts | High | Core innovation: separate definition from execution |
| **Auto-Logging** | Zero-code integration with popular frameworks (sklearn, XGBoost, PyTorch) | High | Requires framework-specific integrations |
| **Hyperparameter Optimization** | Built-in sweeps/search (grid, random, Bayesian) | High | Eliminates need for external Optuna/Ray Tune |
| **Artifact Lineage Tracking** | Understand which dataset/model produced which result | Medium | DAG visualization: Dataset → Run → Model → Evaluation |
| **Multi-Metric Comparison Tables** | Side-by-side comparison of 10+ runs | Medium | Parallel coordinates plots, filterable tables |
| **Collaborative Annotation** | Team members can comment on runs, share insights | Medium | Social layer on top of tracking |
| **Reproducibility Reports** | One-click export of code, env, data versions for a run | Medium | Docker environment capture, git hash, requirements.txt |
| **Advanced Visualization** | Custom charts (confusion matrices, ROC curves, embeddings) | High | Extensible visualization framework |
| **Model Registry Integration** | Push trained models directly to staging/production | High | Lifecycle management beyond experiments |
| **Notebook Integration** | Track Jupyter notebook cells as experiments | Medium | IPython magic commands, cell-level tracking |

**Differentiator Rationale:**
- Pain point: "13 experiments run as individual scripts" = solved by config-driven
- Pain point: "No systematic comparison framework" = solved by multi-metric tables
- Value: Researchers can define experiments declaratively, not imperatively

---

## Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| **Real-Time UI Updates (WebSocket)** | Users want to see training live | Complex state management, breaks on network issues | Poll-based updates every 5-10s is sufficient |
| **Cloud-Only Storage** | "Easy to share with team" | Lock-in, egress costs, no offline work | Local-first with optional cloud sync |
| **All-in-One MLOps Platform** | "We need everything in one place" | Bloated, slow, mediocre at everything | Focused experiment tracking, integrate with best-of-breed tools |
| **Automatic Model Deployment** | "One-click to production" | Deployment is domain-specific, needs custom logic | Export model artifacts, let teams deploy their way |
| **Complex Permission System** | Enterprise wants RBAC | Over-engineered for most teams, maintenance burden | Simple workspace-based sharing |
| **Built-in Compute** | "Run experiments on our infra" | Infrastructure is not core competency, hard to scale | Focus on tracking, integrate with AWS/GCP/Azure |
| **Auto-Recovery from Crashes** | "Don't lose training if it dies" | Complex state serialization, often fragile | Checkpointing is user's responsibility |
| **Monolithic UI** | "Everything in one dashboard" | Slow, cluttered, hard to navigate | Composable UI: separate pages for runs, compare, artifacts |

**Anti-Feature Rationale:**
- Keep the framework lightweight and focused
- Current project is local research - cloud complexity is unnecessary
- Integration > bundling

---

## Feature Dependencies

```
[Run Tracking]
    └──requires──> [Experiment Grouping]
                   └──enhances──> [Search/Filter]

[Configuration-Driven Experiments]
    └──requires──> [Python SDK]
                   └──requires──> [Artifact Storage]

[Auto-Logging]
    ├──requires──> [Python SDK]
    └──enhances──> [Parameter Logging]
                   └──enhances──> [Metric Logging]

[Hyperparameter Optimization]
    ├──requires──> [Run Tracking]
    ├──requires──> [Configuration-Driven Experiments]
    └──enhances──> [Multi-Metric Comparison]

[Artifact Lineage]
    ├──requires──> [Artifact Storage]
    └──enhances──> [Reproducibility Reports]

[Web UI]
    ├──requires──> [Run Tracking]
    ├──requires──> [Search/Filter]
    └──enhances──> [Multi-Metric Comparison]

[Model Registry Integration]
    └──requires──> [Artifact Storage]
```

**Dependency Notes:**

- **Run Tracking requires Experiment Grouping:** Without organization, 1000+ runs are unmanageable
- **Configuration-Driven requires Python SDK & Artifact Storage:** Config must be parsed and executed, outputs stored
- **Auto-Logging enhances Parameter/Metric Logging:** Removes boilerplate, but underlying storage is same
- **Hyperparameter Optimization requires Configuration-Driven:** Can't sweep if experiments are hardcoded in scripts
- **Artifact Lineage enhances Reproducibility:** Knowing data flow enables one-click environment capture
- **Model Registry requires Artifact Storage:** Can't manage models if they're not versioned

---

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

**Core MVP: Replace ad-hoc scripts with tracked experiments**

- [x] **Run Tracking** — Every execution is recorded with start/end time, status
- [x] **Parameter Logging** — Record hyperparameters (lr, epochs, model_type)
- [x] **Metric Logging** — Record metrics (RMSE, R2, loss) over time
- [x] **Artifact Storage** — Save models, predictions, plots
- [x] **Experiment Grouping** — Organize runs by project/experiment
- [x] **Python SDK** — Simple API: `start_run()`, `log_param()`, `log_metric()`, `log_artifact()`
- [x] **CLI Access** — `exp-run config.yaml`, `exp-list`, `exp-compare`
- [x] **Basic Web UI** — Table view of runs, detail view, simple charts
- [x] **Search/Filter** — Filter runs by metric/parameter values

**Why This MVP:**
- Solves core pain: "13 experiments scattered across files"
- Enables: "Track all experiment metadata and results"
- Enables: "Compare experiments systematically"
- Low enough complexity to build quickly

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Configuration-Driven Experiments** — YAML-based experiment definitions
  - **Trigger:** When users write 5+ similar run scripts
- [ ] **Tagging System** — Custom labels for organization
  - **Trigger:** When search/filter becomes cumbersome with only parameters
- [ ] **Artifact Lineage Tracking** — Dataset → Model → Result graph
  - **Trigger:** When users ask "which data produced this model?"
- [ ] **Multi-Metric Comparison Tables** — Side-by-side run comparison
  - **Trigger:** When users compare 5+ runs manually
- [ ] **Auto-Logging** — Framework integrations (sklearn, XGBoost)
  - **Trigger:** When boilerplate `log_param()` calls become tedious

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Hyperparameter Optimization** — Built-in sweeps/search
  - **Why Defer:** Integration complexity, existing tools (Optuna) work well
- [ ] **Advanced Visualization** — Custom charts beyond line plots
  - **Why Defer:** Hard to predict what visualizations researchers need
- [ ] **Model Registry Integration** — Lifecycle management
  - **Why Defer:** Research framework first, production deployment later
- [ ] **Notebook Integration** — Jupyter tracking
  - **Why Defer:** Most training happens in scripts, not notebooks
- [ ] **Reproducibility Reports** — One-click environment export
  - **Why Defer:** Nice-to-have, manual git tracking works initially
- [ ] **Collaborative Annotation** — Comments/sharing
  - **Why Defer:** Single-user focus initially

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Run Tracking | HIGH | Low | **P1** |
| Parameter Logging | HIGH | Low | **P1** |
| Metric Logging | HIGH | Low | **P1** |
| Artifact Storage | HIGH | Medium | **P1** |
| Experiment Grouping | HIGH | Low | **P1** |
| Python SDK | HIGH | Medium | **P1** |
| CLI Access | HIGH | Medium | **P1** |
| Basic Web UI | HIGH | Medium | **P1** |
| Search/Filter | HIGH | Low | **P1** |
| Tagging System | MEDIUM | Low | **P2** |
| Multi-Metric Comparison | MEDIUM | Medium | **P2** |
| Configuration-Driven | HIGH | High | **P2** |
| Auto-Logging | MEDIUM | High | **P2** |
| Artifact Lineage | MEDIUM | High | **P3** |
| Hyperparameter Optimization | HIGH | High | **P3** |
| Advanced Visualization | MEDIUM | High | **P3** |
| Model Registry Integration | LOW | High | **P3** |
| Notebook Integration | LOW | Medium | **P3** |
| Reproducibility Reports | MEDIUM | High | **P3** |
| Collaborative Annotation | LOW | Medium | **P3** |

**Priority key:**
- **P1**: Must have for MVP (launch with these)
- **P2**: Should have, add when possible (v1.x)
- **P3**: Nice to have, future consideration (v2+)

---

## Competitor Feature Analysis

| Feature | MLflow | Weights & Biases | Comet | Our Framework |
|---------|--------|------------------|-------|---------------|
| **Run Tracking** | ✅ Core concept | ✅ Core concept | ✅ Core concept | ✅ Table Stakes |
| **Parameter Logging** | ✅ `log_param()` | ✅ `wandb.config` | ✅ `log_parameters()` | ✅ Table Stakes |
| **Metric Logging** | ✅ `log_metric()` | ✅ `wandb.log()` | ✅ `log_metrics()` | ✅ Table Stakes |
| **Artifact Storage** | ✅ `log_artifact()` | ✅ `wandb.save()` | ✅ `log_asset()` | ✅ Table Stakes |
| **Auto-Logging** | ✅ `autolog()` (15+ frameworks) | ✅ Framework integrations | ✅ Framework integrations | ⏳ v1.x (P2) |
| **Hyperparameter Optimization** | ❌ (requires integration) | ✅ Sweeps (built-in) | ✅ Optimizer (built-in) | ⏳ v2+ (P3) |
| **Configuration-Driven** | ❌ (code-centric) | ❌ (code-centric) | ❌ (code-centric) | ✅ **Differentiator** |
| **Web UI** | ✅ Full-featured | ✅ Full-featured | ✅ Full-featured | ✅ Basic (P1), Enhanced (P2) |
| **Model Registry** | ✅ Built-in | ✅ Built-in | ✅ Built-in | ⏳ v2+ (P3) |
| **Artifact Lineage** | ✅ MLflow Tracking | ✅ Artifacts | ✅ Artifacts | ⏳ v1.x (P2) |
| **Local-First Storage** | ✅ Local files by default | ❌ Cloud-only (mostly) | ❌ Cloud-only (mostly) | ✅ **Differentiator** |

**Key Differentiation Strategy:**
1. **Configuration-Driven Experiments** — None of the competitors focus on YAML/config-based experiment definition
2. **Local-First Design** — MLflow is local-capable but defaults to server; W&B/Comet are cloud-centric
3. **Lightweight & Focused** — Competitors are bloated platforms; we stay focused on experiment tracking

---

## Domain-Specific Features (ML Experimentation)

These features are specific to ML experimentation, not generic software:

### Experiment Definition & Configuration
- **Hyperparameter Templating** — Jinja2-like variable substitution in configs
- **Dataset Aliases** — Refer to datasets by name, not path
- **Model Architecture Registry** — Reusable model definitions

### Execution & Tracking
- **Checkpoint Resume** — Continue interrupted runs from last checkpoint
- **Child Runs** — Nested runs for cross-validation folds, multi-task
- **Run Status States** — Pending, Running, Completed, Failed, Killed

### Results Storage & Retrieval
- **Time-Series Metrics** — Metrics logged over training steps
- **Artifact Versioning** — Multiple artifacts per run with versioning
- **Metadata Queries** — SQL-like filtering on params/metrics

### Comparison & Analysis
- **Parallel Coordinates Plot** — Multi-parameter visualization
- **Metric Correlation** — Understand which parameters affect performance
- **Run Grouping** — Compare runs by tag, experiment, or custom criteria

### Visualization & Reporting
- **Training Curves** — Live metric charts during training
- **Confusion Matrices** — Classification-specific visualization
- **Residual Plots** — Regression-specific visualization (relevant for biomass)
- **Embedding Projections** — High-dimensional visualization (PCA, t-SNE)

### Reproducibility
- **Git Integration** — Auto-capture commit hash, branch
- **Environment Capture** — Python packages, CUDA version, OS
- **Random Seed Tracking** — Record seeds for reproducibility
- **Data Versioning** — Hash or version of training data

---

## Implementation Notes

### Data Model (Core Entities)

```python
Run:
  - id: UUID
  - experiment_id: UUID (FK)
  - name: str
  - status: Enum (pending, running, completed, failed, killed)
  - start_time: datetime
  - end_time: datetime (nullable)
  - tags: Dict[str, str]
  - params: Dict[str, Any]  # Hyperparameters
  - metrics: TimeSeries[key, value, step, timestamp]
  - artifacts: List[Artifact]

Experiment:
  - id: UUID
  - project_id: UUID (FK)
  - name: str
  - description: str
  - created_at: datetime

Artifact:
  - id: UUID
  - run_id: UUID (FK)
  - path: str  # URI or local path
  - type: Enum (model, plot, data, other)
```

### Storage Backend Options

1. **SQLite** (MVP)
   - Single file, zero-setup
   - Sufficient for 10K+ runs
   - Easy migration path to PostgreSQL

2. **PostgreSQL** (Scale-up)
   - Better concurrent access
   - Full-text search on tags/descriptions
   - JSONB for flexible params/metrics

3. **File System** (Artifacts)
   - `artifacts/<run_id>/` directory structure
   - Optional S3/GCS backend for cloud

### API Design (Python SDK)

```python
# Table Stakes API
with start_run(experiment="biomass-prediction", name="resnet18-baseline") as run:
    run.log_params({"lr": 0.001, "epochs": 100})
    run.log_metrics({"train_loss": 0.123, "val_rmse": 14.5}, step=10)
    run.log_artifact("model.pth", type="model")
    run.log_artifact("predictions.csv", type="data")
    run.add_tags({"baseline": "true", "architecture": "resnet18"})

# v1.x: Configuration-Driven
run_experiment("configs/resnet18.yaml")  # Executes config as tracked run

# Query API
runs = search_runs(filter="metrics.val_rmse < 12", order_by="metrics.val_rmse ASC")
compare_runs(run_ids=[...], metrics=["val_rmse", "train_loss"])
```

---

## Sources

### HIGH Confidence (Official Documentation)
- **MLflow Tracking Documentation** - https://mlflow.org/docs/latest/tracking.html
  - Core concepts: Runs, Experiments, Parameters, Metrics, Artifacts
  - Auto-logging, API design, storage backends
- **Weights & Biases Track Guide** - https://docs.wandb.ai/guides/track
  - Run tracking, logging workflow, integrations
- **Comet Experiment Management** - https://www.comet.com/docs/
  - Experiment lifecycle, artifact management, visualization

### MEDIUM Confidence (WebSearch Results)
- "Machine Learning Experimentation Platform: Must-Have Features"
  - Features categorized: Tracking, Reproducibility, Collaboration, MLOps
  - Sources: jfrog.com, kiroframe.com, dagshub.com, viso.ai

### GROUND TRUTH (Project Analysis)
- **Image2Biomass Project Structure** - 29 experiment scripts analyzed
  - Current pain points: scattered results, no comparison, ad-hoc scripts
  - `report.md`: 10 phases, 13 experiments, no systematic tracking
  - Scripts: `train_*.py`, `ensemble_*.py`, `extract_*.py` - all untracked

---

*Feature research for: ML Experimentation Frameworks*
*Researched: 2026-01-17*
*Confidence: HIGH*
