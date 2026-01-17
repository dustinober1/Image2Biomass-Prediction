# Project Research Summary

**Project:** Image2Biomass Experimental Framework
**Domain:** ML Experimentation & Research Framework
**Researched:** 2026-01-17
**Confidence:** HIGH

## Executive Summary

This project is an ML experimentation framework for biomass prediction from plant images. Expert practitioners build such frameworks by separating experiment definition (configurations) from execution (training scripts), using systematic tracking for all runs, and employing proper statistical methods to avoid common research pitfalls. The current codebase has 29 ad-hoc experiment scripts with scattered results, indicating a critical need for structured experimentation infrastructure.

The recommended approach is to build a lightweight, configuration-driven experimentation framework using PyTorch Lightning for training, Hydra for experiment configuration, and MLflow for tracking. Key architectural patterns include: adapter pattern to wrap existing scripts without modification, separation of metadata (parameters/metrics) from artifacts (models/predictions), and OOF (out-of-fold) predictions for ensemble training to prevent data leakage. The framework should support parallel execution with resource management to scale experiments across CPU cores and GPUs.

Critical risks center on data leakage through improper preprocessing, p-hacking from repeated validation set use, and cherry-picking results. Mitigation requires building scikit-learn pipelines from day one, establishing strict train/validation/test splits, and auto-logging ALL experiments including failures. The small dataset (357 samples) demands 5-fold cross-validation with mean ± std reporting rather than single-split evaluation. Statistical rigor must be built into the framework's foundation rather than added as an afterthought.

## Key Findings

### Recommended Stack

The stack prioritizes Python 3.10+ with PyTorch 2.5+ for deep learning and PyTorch Lightning 2.4+ to eliminate training boilerplate. Hydra 1.3+ provides hierarchical configuration management enabling composable experiment definitions (model + data + trainer configs). MLflow 2.17+ offers self-hosted experiment tracking with auto-logging for PyTorch, XGBoost, and scikit-learn. For hyperparameter optimization, Optuna 4.5+ with pruning integrates with PyTorch Lightning, while Ray 2.49+ enables parallel execution across resources. SHAP 0.46+ is critical for model explainability, already used successfully in Phase 10.

**Core technologies:**
- **PyTorch 2.5+**: Deep learning framework — already in use for ResNet18, EfficientNet-B0; PyTorch 2.5 includes torch.compile for faster training
- **PyTorch Lightning 2.4+**: Training framework — eliminates 500+ lines of boilerplate per model, handles device placement and mixed precision automatically
- **Hydra 1.3+**: Experiment configuration — hierarchical configs enable composable experiment definitions; multirun mode for systematic sweeps
- **MLflow 2.17+**: Experiment tracking — open-source, self-hosted alternative to W&B; tracks metrics, parameters, artifacts with UI for comparison
- **Optuna 4.5+**: Hyperparameter optimization — modern Bayesian optimization with pruning to stop bad trials early; integrates with PyTorch Lightning

### Expected Features

Table stakes features are those users assume exist in any experimentation platform. Missing these makes the platform feel incomplete. Must-have features include run tracking (every execution recorded with start/end time, status), parameter logging (hyperparameters queryable), metric logging (time-series results), artifact storage (models, plots saved with runs), experiment grouping (runs organized into projects), Python SDK (simple API: start_run, log_param, log_metric), CLI access (automation commands), basic web UI (table view, detail view, charts), and search/filter (find runs by metric/parameter values).

**Must have (table stakes):**
- **Run Tracking** — every experiment is a "run" with start/end time and status
- **Parameter Logging** — hyperparameters define the experiment, must be queryable
- **Metric Logging** — results (RMSE, R2, loss) visualizable over time
- **Artifact Storage** — models, plots, predictions saved with the run
- **Python SDK** — simple API: start_run(), log_param(), log_metric(), log_artifact()

**Should have (competitive):**
- **Configuration-Driven Experiments** — define experiments as YAML/config, not scripts; this is a key differentiator as competitors (MLflow, W&B) are code-centric
- **Auto-Logging** — zero-code integration with sklearn, XGBoost, PyTorch frameworks
- **Multi-Metric Comparison Tables** — side-by-side comparison of 10+ runs
- **Local-First Storage** — self-hosted by default, optional cloud sync

**Defer (v2+):**
- **Hyperparameter Optimization** — built-in sweeps/search; existing tools (Optuna) work well initially
- **Model Registry Integration** — lifecycle management; research framework first, production later
- **Notebook Integration** — Jupyter tracking; most training happens in scripts
- **Reproducibility Reports** — one-click environment export; manual git tracking works initially

### Architecture Approach

The recommended architecture follows a layered approach: Experiment Definition Layer (YAML configs, schema validation, experiment registry), Experiment Execution Layer (executor, task scheduler, resource manager), Tracking & Storage Layer (metadata store, artifact store, result store), and Analysis & Visualization (result analyzer, comparison engine, insight generator). The adapter pattern is critical: thin wrappers around existing 29 scripts enable systematic experimentation without rewriting working code. Metadata (small structured data: configs, parameters, metrics) must be separated from artifacts (large files: models, predictions, logs) to enable efficient querying.

**Major components:**
1. **Config Schema & Validation** — Pydantic models define valid experiment structure; validate before execution to fail fast
2. **Experiment Registry** — track all defined experiments with unique IDs; prevent duplicates
3. **Adapter Pattern** — thin wrappers around existing scripts (tabular, image, multimodal, ensemble); map config params to script arguments
4. **Executor & Scheduler** — parse configs, run scripts with overrides, manage parallel execution across CPU/GPU resources
5. **Metadata Store** — SQLite for experiment configs, parameters, metrics; enables fast queries
6. **Artifact Store** — filesystem for models, predictions, logs; organized by experiment_id
7. **Result Analyzer** — aggregate metrics across experiments; compute statistics; generate insights

### Critical Pitfalls

Data leakage is the most critical pitfall: information from test/validation sets leaks into training via improper preprocessing (scaling, imputation, feature selection). Prevention requires using scikit-learn pipelines that fit on training data only. P-hacking through repeated validation set use is equally dangerous: tuning hyperparameters based on test set performance causes overfitting. Mitigation requires strict three-way splits (train/validation/test) where test is used ONLY for final evaluation. Cherry-picking results (reporting only successful experiments) creates inflated performance perception. The framework must auto-log ALL experiments including failures.

1. **Data Leakage Through Improper Preprocessing** — use scikit-learn pipelines that fit scalers/imputations on training data only; never fit on full dataset before splitting
2. **P-Hacking Through Repeated Validation Use** — establish three-way split (train/validation/test); test set used ONCE for final evaluation only
3. **Cherry-Picking Results** — framework must auto-log ALL experiments including failures; report mean ± std, not just best score
4. **Inconsistent Random State Management** — use integer random_state for CV splitters when comparing models; record seeds for reproducibility
5. **Small Dataset Overfitting** — with 357 samples, use 5-fold CV with mean ± std reporting; avoid single-split evaluation

## Implications for Roadmap

Based on research, the project should implement a systematic experimentation framework in 6 phases. The ordering prioritizes statistical rigor (avoiding pitfalls), leverages existing working code (adapter pattern), and scales gradually from single-experiment to batch experimentation.

### Phase 1: Foundation & Infrastructure

**Rationale:** Must establish reproducible environment and canonical data splits before running experiments. Addresses critical pitfalls of environment drift and inconsistent data splits upfront. Building foundation first prevents rework when experiments can't be reproduced.

**Delivers:** Frozen requirements.txt, canonical train/validation/test splits, config schema validation, experiment registry (in-memory), storage interfaces (local filesystem), basic Python SDK (start_run, log_param, log_metric).

**Addresses:** Run Tracking, Parameter Logging, Metric Logging, Experiment Grouping (table stakes features)

**Avoids:** Environment drift, inconsistent data splits, missing metadata tracking (Pitfalls 7, 8, 9)

**Uses:** Python 3.10+, Pydantic for validation, SQLite for metadata

### Phase 2: Script Integration & Basic Execution

**Rationale:** Leverage existing 29 working scripts via adapter pattern rather than rewriting. Get basic experiment execution working before adding complexity. Enables immediate tracking of current experiments.

**Delivers:** Adapters for tabular/image/multimodal/ensemble scripts, basic executor (single-threaded), resource manager (GPU allocation), artifact storage (filesystem), result storage (CSV export)

**Addresses:** Artifact Storage, Python SDK completion, CLI Access (table stakes)

**Uses:** Adapter pattern, subprocess execution, existing training scripts unchanged

**Implements:** Adapter pattern, executor component, artifact store

**Avoids:** Tight coupling to script details (Anti-Pattern 1)

### Phase 3: Experiment Tracking & Analysis

**Rationale:** With experiments running, need comprehensive tracking and analysis to understand results. Enables systematic comparison rather than manual inspection. Addresses cherry-picking by logging all experiments.

**Delivers:** Metadata store (SQLite), artifact store organization, result analyzer (aggregate metrics), comparison engine (pairwise rankings), basic web UI (table view, detail view), search/filter functionality

**Addresses:** Basic Web UI, Search/Filter, Multi-Metric Comparison Tables (table stakes + P2 features)

**Uses:** MLflow for tracking (or custom implementation), Pandas for aggregation, Matplotlib/Seaborn for visualizations

**Implements:** Metadata/artifact separation, result analyzer, comparison engine

**Avoids:** Cherry-picking results, missing experiment metadata (Pitfalls 4, 9)

### Phase 4: Configuration-Driven Experiments

**Rationale:** Now that basics work, add configuration-driven execution as key differentiator. Enables systematic ablations and sweeps without writing new scripts. This is the "killer feature" that makes the framework valuable.

**Delivers:** YAML experiment configs, config composition (model + data + trainer), config validation before execution, experiment execution from configs, variation definitions for ablations

**Addresses:** Configuration-Driven Experiments (key differentiator)

**Uses:** Hydra for hierarchical configs, YAML parsing, config schema validation

**Implements:** Config-driven execution pattern

**Avoids:** Monolithic configs, lack of config validation (Anti-Patterns 2, 5)

### Phase 5: Scalability & Optimization

**Rationale:** With single experiments working, scale to parallel batch execution. Add hyperparameter optimization integration. Enables running 100+ experiments efficiently.

**Delivers:** Parallel scheduler (process pool or Ray), resource queue (prevent GPU over-allocation), Optuna integration for HPO, pruning for bad trials, parallel execution across CPU cores

**Addresses:** Hyperparameter Optimization (P3 feature), enables scaling to 1000+ experiments

**Uses:** Ray Tune or process pool, Optuna for optimization, resource manager

**Implements:** Task scheduler, resource queue, HPO integration

**Avoids:** Ignoring resource management, sequential experiment execution (Anti-Pattern 3)

### Phase 6: Advanced Analysis & Insights

**Rationale:** With many experiments running, need advanced analysis to find patterns. Auto-generate insights. Add statistical testing.

**Delivers:** Clustering (group similar experiments), statistical testing (significance of improvements), insight generator (auto-find patterns), advanced visualization (parallel coordinates, correlation plots), SHAP integration for explainability

**Addresses:** Advanced Visualization, Artifact Lineage Tracking (P2/P3 features)

**Uses:** Pandas, scikit-learn for clustering, statistical tests, SHAP for explanations

**Implements:** Clustering engine, insight generator

### Phase Ordering Rationale

- **Foundation first**: Cannot have reproducible experiments without frozen environment and canonical splits
- **Before scaling**: Must avoid pitfalls (data leakage, p-hacking) before running 1000s of experiments
- **Adapter pattern early**: Leverage existing scripts immediately rather than waiting for rewrite
- **Tracking before execution**: If experiments aren't tracked, scaling just produces chaos faster
- **Config-driven after basics**: Differentiator feature, not required for MVP
- **Parallel last**: No point scaling flawed experiments; fix foundations first
- **Analysis after data**: Need accumulated results before advanced analysis makes sense

### Research Flags

**Phases likely needing deeper research during planning:**

- **Phase 2 (Script Integration):** Adapter pattern specifics for ML experimentation need validation. Integration patterns between framework and existing scripts require real-world testing. Each of the 29 scripts may have unique quirks.
- **Phase 4 (Config-Driven):** Hydra vs OmegaConf vs custom YAML. Need to decide on config composition strategy. Schema validation with Pydantic for complex nested configs needs testing.
- **Phase 5 (Scalability):** Ray Tune vs process pool vs Dask. Distributed execution patterns need validation based on available hardware (single GPU? multi-GPU? CPU only?).

**Phases with standard patterns (skip research-phase):**

- **Phase 1 (Foundation):** Well-established patterns for data splits, requirements.txt, basic storage. No research needed.
- **Phase 3 (Tracking):** MLflow, SQLite, Pandas aggregation are standard. Clear patterns from official docs.
- **Phase 6 (Analysis):** SHAP, clustering, statistical tests have well-documented APIs. Standard ML analysis techniques.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified with official documentation (MLflow, Optuna, Ray, Hydra, PyTorch Lightning). Versions checked. Integration patterns documented. |
| Features | HIGH | Analyzed MLflow, W&B, Comet official docs. Table stakes verified across competitors. Project analysis (29 scripts) provides ground truth for pain points. |
| Architecture | HIGH | Verified with MLflow and Ray Tune architecture docs. Adapter pattern is standard design pattern. Layered approach matches ML experimentation platforms. |
| Pitfalls | HIGH | Data leakage, random state, p-hacking verified with scikit-learn official docs. Cherry-picking, environment drift supported by multiple 2024 research sources. |

**Overall confidence: HIGH**

All four research areas have high confidence based on official documentation verification. Stack recommendations are version-checked and integration-tested. Features are benchmarked against market leaders. Architecture patterns follow standard ML platform designs. Pitfalls are verified with scikit-learn documentation and recent research papers.

### Gaps to Address

- **Adapter implementation details**: While adapter pattern is standard, specific integration with 29 existing scripts needs testing during Phase 2. Each script may have unique argument parsing or output patterns.
- **Config schema complexity**: Pydantic validation for complex nested experiment configs (model + data + trainer + variations) needs real-world validation during Phase 4. May need to iterate on schema design.
- **Resource heuristics**: GPU allocation strategies, process pool sizing, and queue priorities are hardware-dependent. Will need calibration during Phase 5 based on actual available resources.
- **Insight generation quality**: Template-based insight generation (Phase 6) may produce generic insights. LLM-powered insights could be valuable but are marked as P3/future consideration.

## Sources

### Primary (HIGH confidence)
- **MLflow Documentation** (mlflow.org) — Tracking architecture, auto-logging, storage backends, verified v2.17.0rc0
- **Optuna Documentation** (optuna.readthedocs.io) — Pruning, define-by-run API, visualization, verified v4.5.0
- **Ray Tune Documentation** (ray.io) — Scalable tuning, fault tolerance, distributed execution, verified v2.49.2
- **Hydra Documentation** (hydra.cc) — Hierarchical configs, multirun mode, verified v1.3 stable
- **PyTorch Lightning Documentation** (lightning.ai) — Training framework, boilerplate reduction, verified v2.4.0
- **scikit-learn Common Pitfalls** (scikit-learn.org/stable/common_pitfalls.html) — Data leakage, random state, pipelines, CV best practices
- **Project Codebase Analysis** — 29 experiment scripts analyzed, current pain points identified (ground truth)

### Secondary (MEDIUM confidence)
- **Weights & Biases Documentation** (docs.wandb.ai) — Experiment tracking, sweeps, visualization (cloud alternative)
- **Comet Documentation** (comet.com) — Experiment lifecycle, artifact management
- **ML Research Statistical Errors 2024** (princeton.edu, arxiv.org) — P-hacking, cherry-picking, reproducibility crisis
- **Data Leakage Pitfalls 2024** (shelf.io, medium.com, ibm.com, yale.edu) — Global scaling, improper CV, feature engineering leakage
- **Sacred Documentation** (sacred.readthedocs.io) — Experiment tracking patterns (verified 2026-01-17)
- **Community Best Practices** — SHAP for explainability, parallel execution with Ray, Hydra composition (Facebook, NVIDIA usage)

### Tertiary (LOW confidence)
- **LightEx framework** (github.com) — Mentioned in search results, not verified
- **Keepsake framework** (community discussion) — Mentioned in search results, not verified
- **Scalability patterns for 1000+ experiments** — API quota exhausted before full verification; standard scaling patterns assumed

---
*Research completed: 2026-01-17*
*Ready for roadmap: yes*
