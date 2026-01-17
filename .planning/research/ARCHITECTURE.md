# Architecture Research

**Domain:** ML Experimentation Framework
**Researched:** 2026-01-17
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Experiment Definition Layer                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ YAML Configs │  │ Config Schema│  │ Experiment   │              │
│  │              │  │ Validation   │  │ Registry     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼───────────────────┼─────────────────────┘
          │                  │                   │
          ↓                  ↓                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     Experiment Execution Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Experiment   │  │ Task Queue / │  │ Resource     │              │
│  │ Executor     │  │ Scheduler    │  │ Manager      │              │
│  │              │  │              │  │              │              │
│  │ - Parse cfg  │  │ - Parallel   │  │ - GPU alloc  │              │
│  │ - Wrap script│  │ - Priority   │  │ - CPU cores  │              │
│  │ - Run trial  │  │ - Dependencies│ │ - Fault tol  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼───────────────────┼─────────────────────┘
          │                  │                   │
          ↓                  ↓                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Tracking & Storage Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Metadata     │  │ Artifact     │  │ Result       │              │
│  │ Store        │  │ Store        │  │ Store        │              │
│  │              │  │              │  │              │              │
│  │ - Exp config │  │ - Models     │  │ - Metrics    │              │
│  │ - Parameters │  │ - OOF preds  │  │ - CSVs       │              │
│  │ - Status     │  │ - Logs       │  │ - Plots      │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼───────────────────┼─────────────────────┘
          │                  │                   │
          ↓                  ↓                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Analysis & Visualization                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Result       │  │ Comparison   │  │ Insight      │              │
│  │ Analyzer     │  │ Engine       │  │ Generator    │              │
│  │              │  │              │  │              │              │
│  │ - Aggregate  │  │ - Pairwise   │  │ - Auto-find  │              │
│  │ - Cluster    │  │ - Ranking    │  │ patterns     │              │
│  │ - Stat tests │  │ - Filter     │  │ - Reports    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Config Schema** | Define valid experiment structure, validate inputs | Pydantic models, JSON Schema |
| **Experiment Registry** | Track all defined experiments, prevent duplicates | SQLite/PostgreSQL, in-memory dict |
| **Experiment Executor** | Parse config, wrap training scripts, execute trials | Python subprocess, jobliblib |
| **Task Scheduler** | Manage parallel execution, handle dependencies | Process pool, Ray, Dask |
| **Resource Manager** | Allocate GPU/CPU, prevent resource conflicts | CUDA management, psutil |
| **Metadata Store** | Persist experiment configs, parameters, status | SQLite, PostgreSQL, file system |
| **Artifact Store** | Store large files (models, predictions, logs) | File system with organized paths |
| **Result Store** | Store metrics, evaluation results | CSV, Parquet, database |
| **Result Analyzer** | Aggregate results, compute statistics | Pandas, NumPy |
| **Comparison Engine** | Compare experiments, rank, filter | Pandas, SQL queries |
| **Insight Generator** | Auto-generate findings from result patterns | Templating, LLM integration |

## Recommended Project Structure

```
src/
├── framework/              # Core experimentation framework
│   ├── config/            # Configuration management
│   │   ├── schema.py      # Pydantic models for validation
│   │   ├── loader.py      # YAML config loader
│   │   └── templates/     # Example experiment configs
│   ├── registry/          # Experiment registry
│   │   ├── registry.py    # Track defined experiments
│   │   └── storage.py     # Backend storage interface
│   ├── executor/          # Experiment execution
│   │   ├── runner.py      # Wrap and run training scripts
│   │   ├── scheduler.py   # Parallel execution manager
│   │   └── resources.py   # GPU/CPU allocation
│   ├── tracking/          # Experiment tracking
│   │   ├── metadata.py    # Log configs, params, status
│   │   ├── artifacts.py   # Save models, predictions
│   │   └── metrics.py     # Store evaluation results
│   ├── storage/           # Storage backends
│   │   ├── backend.py     # Abstract storage interface
│   │   ├── local.py       # Local filesystem storage
│   │   └── database.py    # SQLite/PostgreSQL backend
│   └── analysis/          # Result analysis
│       ├── aggregator.py  # Aggregate results across experiments
│       ├── comparison.py  # Compare experiments
│       ├── clustering.py  # Group similar experiments
│       └── insights.py    # Generate findings
├── adapters/              # Wrappers for existing scripts
│   ├── tabular.py         # Adapter for tabular models
│   ├── image.py           # Adapter for image models
│   ├── multimodal.py      # Adapter for multimodal models
│   └── ensemble.py        # Adapter for ensemble scripts
├── scripts/               # Existing training scripts (unchanged)
│   ├── train_tabular_baseline.py
│   ├── train_stacking_meta.py
│   └── ... (27 other scripts)
├── experiments/           # Experiment configurations
│   ├── ablations/         # Feature ablation experiments
│   ├── models/            # Model comparison experiments
│   └── ensembles/         # Ensemble strategy experiments
└── outputs/               # Experiment outputs
    ├── runs/              # Per-run directories
    │   ├── exp_001/       # Config, logs, artifacts
    │   └── exp_002/
    ├── results/           # Aggregated results
    │   ├── metrics.csv    # All experiment metrics
    │   └── comparisons/   # Comparison outputs
    └── insights/          # Generated insights
        └── reports/       # Auto-generated reports
```

### Structure Rationale

- **framework/**: Core experimentation logic, independent of specific ML tasks
  - Separation of concerns: config, execution, tracking, analysis are distinct
  - Pluggable storage: can swap local files for database later
- **adapters/**: Thin wrappers around existing scripts, no logic changes
  - Preserves working code while enabling systematic experimentation
  - Each adapter maps config params to script arguments
- **scripts/**: Keep existing training scripts unchanged
  - Reduces risk of breaking working experiments
  - Allows gradual migration to framework approach
- **experiments/**: YAML configs defining experiments to run
  - Human-readable, version-controlled experiment definitions
  - Easy to create ablations by varying single params
- **outputs/**: Organized by experiment run, then aggregated
  - Per-run isolation prevents conflicts
  - Aggregated results enable analysis across runs

## Architectural Patterns

### Pattern 1: Configuration-Driven Execution

**What:** Experiments defined as YAML configs, executed by framework
**When to use:** Need systematic variation of parameters across many runs
**Trade-offs:**
- Pros: Reproducible, version-controllable, parallelizable
- Cons: Learning curve for YAML, less flexible than ad-hoc scripts

**Example:**
```yaml
# experiments/ablations/feature_ablation.yaml
experiment:
  name: "feature_ablation_grid_features"
  description: "Test with/without grid features"
  base_script: "scripts/train_ridge_advanced.py"

  # Base configuration
  config:
    data:
      oof_tabular: "models/stacking/tabular/oof_tabular.csv"
      oof_kmeans: "models/stacking/kmeans/oof_kmeans.csv"
      oof_effnet: "models/stacking/effnet/oof_effnet.csv"
    model:
      type: "Ridge"
      alpha: 1.0

  # Variations to run
  variations:
    - name: "with_grid_features"
      data:
        features_grid: "models/features_grid/features_grid_train.csv"
    - name: "without_grid_features"
      data:
        features_grid: null

  # Execution settings
  execution:
    parallel: true
    num_workers: 2
    resources:
      num_gpus: 0
      num_cpus: 4
```

### Pattern 2: Adapter Pattern for Script Integration

**What:** Thin adapter classes wrap existing scripts, exposing uniform interface
**When to use:** Have working scripts that need systematic experimentation
**Trade-offs:**
- Pros: No script changes, gradual migration, testable adapters
- Cons: Adapter maintenance overhead, indirect execution

**Example:**
```python
# adapters/tabular.py
class TabularAdapter:
    """Adapter for tabular baseline training script"""

    def __init__(self, config: dict):
        self.config = config
        self.script_path = "scripts/train_tabular_baseline.py"

    def run(self, experiment_id: str):
        """Execute the script with config overrides"""
        import subprocess

        # Extract relevant params from config
        learning_rate = self.config.get('training', {}).get('learning_rate', 0.05)
        max_depth = self.config.get('training', {}).get('max_depth', 5)

        # Build command line args
        cmd = [
            'python', self.script_path,
            '--learning_rate', str(learning_rate),
            '--max_depth', str(max_depth),
            '--output_dir', f'outputs/runs/{experiment_id}'
        ]

        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def validate_config(self):
        """Check if config has required fields"""
        required = ['data', 'model']
        return all(k in self.config for k in required)
```

### Pattern 3: Metadata-Artifact Separation

**What:** Split small structured data (metadata) from large files (artifacts)
**When to use:** Storing experiment results, models, predictions
**Trade-offs:**
- Pros: Fast metadata queries, efficient storage, clear separation
- Cons: Need to maintain two storage systems, linkage complexity

**Example:**
```python
# tracking/metadata.py
class MetadataStore:
    """Store experiment configs, parameters, metrics"""

    def log_experiment(self, exp_id: str, config: dict):
        """Store experiment config in metadata DB"""
        self.db.execute(
            "INSERT INTO experiments (id, config, status) VALUES (?, ?, ?)",
            (exp_id, json.dumps(config), 'running')
        )

    def log_metrics(self, exp_id: str, metrics: dict):
        """Store evaluation metrics"""
        self.db.execute(
            "INSERT INTO metrics (exp_id, metrics) VALUES (?, ?)",
            (exp_id, json.dumps(metrics))
        )

# tracking/artifacts.py
class ArtifactStore:
    """Store large files: models, predictions, logs"""

    def save_model(self, exp_id: str, model_path: str):
        """Copy trained model to artifact store"""
        artifact_path = self.base_path / exp_id / 'model.pkl'
        shutil.copy(model_path, artifact_path)
        return str(artifact_path)

    def save_predictions(self, exp_id: str, preds: np.ndarray):
        """Save predictions as CSV in artifact store"""
        artifact_path = self.base_path / exp_id / 'predictions.csv'
        pd.DataFrame(preds).to_csv(artifact_path, index=False)
        return str(artifact_path)
```

## Data Flow

### Experiment Execution Flow

```
[YAML Config]
    ↓
[Config Validator] → (validate schema)
    ↓
[Experiment Registry] → (register experiment, assign ID)
    ↓
[Adapter Selector] → (choose script wrapper based on config.model.type)
    ↓
[Resource Manager] → (allocate GPU/CPU)
    ↓
[Executor] → (run script with config params)
    ↓                ↕ (log progress)
[Metadata Store] ← (track status, timestamps)
    ↓
[Artifact Store] ← (save models, predictions, logs)
    ↓
[Result Store] ← (save metrics to CSV/database)
    ↓
[Analysis Engine] ← (triggered on completion)
```

### Result Analysis Flow

```
[Experiment Completes]
    ↓
[Result Store Updated]
    ↓ (query)
[Result Analyzer] → (aggregate metrics across experiments)
    ↓
[Comparison Engine] → (pairwise comparisons, rankings)
    ↓
[Clustering Engine] → (group similar experiments)
    ↓
[Insight Generator] → (auto-generate findings)
    ↓
[Report] → (markdown/HTML with tables, plots)
```

### Key Data Flows

1. **Config → Execution:** YAML config parsed, validated, passed to adapter, adapter runs script with overrides
2. **Execution → Tracking:** Script outputs captured by executor, metadata logged to store, artifacts saved to disk
3. **Results → Analysis:** Aggregated results queried, comparisons computed, insights generated

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1-100 experiments | Local filesystem storage, process pool for parallel execution, SQLite for metadata |
| 100-1000 experiments | Add PostgreSQL for metadata, organize artifacts by date/experiment type, basic resource queuing |
| 1000+ experiments | Distributed execution (Ray/Dask), object storage for artifacts (S3), experiment indexing, result caching |

### Scaling Priorities

1. **First bottleneck:** Parallel execution — scripts already run independently, just need process pool or task queue
2. **Second bottleneck:** Storage I/O — many experiments writing artifacts, need organized directory structure and async writes
3. **Third bottleneck:** Result queries — aggregating across 1000s of experiments, need database indexing and columnar storage

## Anti-Patterns

### Anti-Pattern 1: Tight Coupling to Script Details

**What people do:** Build framework that requires rewriting all training scripts
**Why it's wrong:** High risk, breaks working code, slow migration
**Do this instead:** Use adapter pattern to wrap existing scripts, keep them unchanged

### Anti-Pattern 2: Monolithic Config Files

**What people do:** Put all experiment variations in one massive YAML
**Why it's wrong:** Hard to maintain, difficult to version control specific experiments
**Do this instead:** Modular configs with inheritance, one file per experiment group

### Anti-Pattern 3: Ignoring Resource Management

**What people do:** Launch all experiments in parallel without GPU/CPU limits
**Why it's wrong:** System crashes, slow execution due to resource contention
**Do this instead:** Implement resource manager with queue, allocate GPUs explicitly

### Anti-Pattern 4: Metadata in Artifact Store

**What people do:** Store configs and metrics as JSON files alongside models
**Why it's wrong:** Slow queries, can't aggregate efficiently, no ad-hoc analysis
**Do this instead:** Separate metadata (database/CSV) from artifacts (files), query metadata for analysis

### Anti-Pattern 5: No Config Validation

**What people do:** Load YAML directly and pass to scripts without validation
**Why it's wrong:** Cryptic errors at runtime, hard to debug, invalid experiments waste resources
**Do this instead:** Define schema with Pydantic, validate before execution, fail fast

## Integration Points

### External Libraries

| Library | Integration Pattern | Notes |
|---------|---------------------|-------|
| **MLflow** | Optional tracking backend | Use instead of custom MetadataStore if team already uses it |
| **Ray Tune** | Advanced scheduler | Replace process pool with Ray for distributed execution |
| **Hydra** | Config management | Consider for hierarchical configs if team prefers it over YAML |
| **Weights & Biases** | Cloud tracking | Use for remote teams, replaces local metadata/artifact stores |
| **Pandas** | Result analysis | Standard for aggregating and comparing experiment results |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Framework ↔ Adapters** | Direct Python calls | Adapters are thin wrappers, minimal interface |
| **Executor ↔ Storage** | Async write operations | Don't block execution on I/O, queue writes |
| **Analysis ↔ Storage** | Read-only queries | Analysis layer doesn't modify experiment data |
| **Config Schema ↔ Executor** | Validated dict | Schema ensures executor receives valid config |

## Build Order (Dependencies)

```
Phase 1: Foundation (can build in parallel)
  ├── Config Schema & Validation
  ├── Experiment Registry (in-memory initially)
  └── Storage Interfaces (local filesystem)

Phase 2: Execution (depends on Phase 1)
  ├── Adapters for existing scripts
  ├── Resource Manager (basic GPU allocation)
  └── Executor (single-threaded initially)

Phase 3: Tracking (depends on Phase 1)
  ├── Metadata Store (SQLite)
  ├── Artifact Store (filesystem)
  └── Result Store (CSV export)

Phase 4: Analysis (depends on Phase 3)
  ├── Result Analyzer (aggregate metrics)
  ├── Comparison Engine (pairwise comparisons)
  └── Insight Generator (template-based reports)

Phase 5: Scalability (depends on Phase 2)
  ├── Parallel Scheduler (process pool)
  ├── Resource Queue (prevent over-allocation)
  └── Optional: Ray integration for distributed

Phase 6: Advanced Analysis (depends on Phase 4)
  ├── Clustering (group similar experiments)
  ├── Statistical Testing (significance of improvements)
  └── Optional: LLM-powered insight generation
```

### Recommended Phase Ordering

1. **Start with Phase 1-2:** Get basic experiment execution working with existing scripts
2. **Add Phase 3:** Proper tracking and storage for results
3. **Implement Phase 4:** Basic analysis to understand results
4. **Scale with Phase 5:** Parallel execution for batch experiments
5. **Enhance with Phase 6:** Advanced analysis as experiment count grows

## Sources

**HIGH Confidence (Official Documentation):**
- MLflow Tracking Architecture (mlflow.org) — Verified 2026-01-17
- Ray Tune Architecture (ray.io) — Verified 2026-01-17
- Sacred Documentation (sacred.readthedocs.io) — Verified 2026-01-17

**MEDIUM Confidence (Multiple Source Agreement):**
- Experiment tracking system components (neptune.ai, medium.com, jfrog.com) — Verified 2026-01-17
- ML experimentation framework trends 2026 (kernshell.com, sganalytics.com, medium.com) — Verified 2026-01-17
- Hydra configuration framework (multiple sources) — Verified 2026-01-17

**LOW Confidence (Single Source):**
- LightEx framework (github.com) — Not verified, mentioned in search results
- Keepsake framework (community discussion) — Not verified, mentioned in search results

**Gaps:**
- Specific patterns for result clustering and insight generation were not well-documented in searches
- Integration patterns between experiment tracking and existing ML codebases require validation
- Best practices for adapter pattern in ML experimentation need real-world validation

---
*Architecture research for: ML Experimentation Framework*
*Researched: 2026-01-17*
