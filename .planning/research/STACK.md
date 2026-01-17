# Technology Stack

**Project:** Image2Biomass Experimental Framework
**Domain:** ML Experimentation & Research Framework
**Researched:** 2025-01-17
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.10+ | Core language | Project already uses Python; 3.10+ provides type hints, pattern matching, and better performance. All ML experimentation tools support Python 3.10+. |
| **PyTorch** | 2.5+ | Deep learning framework | Already in use for CNN models (ResNet18, EfficientNet-B0). PyTorch 2.5 includes torch.compile for faster training and better distributed execution. |
| **PyTorch Lightning** | 2.4+ | Training framework | Eliminates boilerplate, handles device placement, gradient accumulation, and mixed precision. Critical for running many experiments systematically without code duplication. |
| **Hydra** | 1.3+ | Experiment configuration | Hierarchical configuration management enables composable experiment definitions (model config + optimizer config + data config). Multirun mode for systematic sweeps. Industry standard for ML research. |
| **MLflow** | 2.17+ | Experiment tracking | Open-source, self-hosted alternative to Weights & Biases. Tracks metrics, parameters, artifacts, and models. UI for comparing experiments. Supports PyTorch, scikit-learn, and custom metrics. |

### Experimentation Infrastructure

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Optuna** | 4.5+ | Hyperparameter optimization | Modern, efficient optimization with pruning (stop bad trials early). Define-by-run API matches Python code naturally. Integrates with PyTorch, PyTorch Lightning, and MLflow. Visualization dashboard included. |
| **Ray** | 2.49+ | Parallel execution | Scale experiments across CPU cores and GPUs. Ray Tune for hyperparameter search integration with Optuna. Fault tolerance for long-running experiment batches. |
| **Weights & Biases** | Latest (optional) | Cloud experiment tracking | Best-in-class visualization and collaboration features. Use if cloud-based tracking is preferred over self-hosted MLflow. Excellent for experiment comparison and sharing results. |
| **DVC** | 3.x (optional) | Data & pipeline versioning | Version datasets and model artifacts. Pipeline orchestration for reproducible experiments. Use if need to track data versions or complex multi-stage pipelines. |

### Model Analysis & Visualization

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **SHAP** | 0.46+ | Model explainability | Already used in Phase 10. SHAP values show feature importance and interaction effects. Works with tree models (XGBoost) and deep models (DeepSHAP). Critical for understanding "why" predictions work. |
| **Matplotlib** | 3.8+ | Base visualization | Standard Python plotting. Required for custom plots, error analysis, and publication-quality figures. |
| **Seaborn** | 0.13+ | Statistical visualization | Higher-level interface for statistical plots (heatmaps, clustermaps, distribution plots). Ideal for analyzing experiment results and error patterns. |
| **Plotly** | 5.20+ | Interactive visualization | Interactive plots for experiment exploration. Integrates with MLflow and W&B for dashboards. Use for exploratory analysis, not static reporting. |
| **Pandas** | 2.2+ | Data manipulation | Already in use. Essential for analyzing experiment results, comparing models, and aggregating metrics across folds. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **scikit-learn** | 1.5+ | Traditional ML | Already in use for Ridge, quantile regression, and CV splits. Required for meta-learners, preprocessing, and metrics (RMSE, R2, MAE). |
| **XGBoost** | 2.1+ | Gradient boosting | Already in use for tabular baseline and meta-learners. Best-in-class for tabular data. Use for ablations on metadata-only models. |
| **Albumentations** | 1.4+ | Data augmentation | Fast image augmentation for PyTorch. Use when experimenting with augmentation strategies (TTA variants, random crops, color jitter). |
| **Rich** | 13.7+ | Console output | Beautiful terminal output for experiment progress. Use for printing experiment results, tables, and progress bars in batch runs. |
| **Typer** | 0.12+ | CLI framework | Type-safe CLI for experiment scripts. Use to create commands like `python run_experiment.py --config experiments/xgboost_ablation.yaml`. |

## Installation

```bash
# Core ML & experimentation
pip install torch>=2.5.0 torchvision>=0.20.0 pytorch-lightning>=2.4.0
pip install hydra-core>=1.3.0 optuna>=4.5.0 ray[tune]>=2.49.0
pip install mlflow>=2.17.0

# Model analysis & visualization
pip install shap>=0.46.0 matplotlib>=3.8.0 seaborn>=0.13.0 plotly>=5.20.0
pip install pandas>=2.2.0 numpy>=1.24.0

# Supporting libraries
pip install scikit-learn>=1.5.0 xgboost>=2.1.0
pip install albumentations>=1.4.0 rich>=13.7.0 typer>=0.12.0

# Optional (cloud tracking)
pip install wandb

# Optional (data versioning)
pip install dvc
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **MLflow** | Weights & Biases | Use W&B if cloud-based collaboration is critical or if superior visualization is worth the hosted cost. W&B has better UI but MLflow is free and self-hosted. |
| **Optuna** | Ray Tune (built-in optimizer) | Use Ray Tune alone if already using Ray heavily and want unified framework. However, Optuna has better pruning algorithms and visualization. |
| **Hydra** | OmegaConf (standalone) | Use OmegaConf alone if Hydra's composition and multirun features aren't needed. Hydra provides better CLI and sweep automation. |
| **PyTorch Lightning** | PyTorch (vanilla) | Use vanilla PyTorch only if experiments are extremely simple. Lightning eliminates 500+ lines of boilerplate per model and ensures reproducibility. |
| **SHAP** | LIME or Captum | Use LIME for local interpretability on any model. Use Captum for PyTorch-specific attribution. SHAP provides unified framework for both global and local explanations. |
| **Matplotlib/Seaborn** | Plotly (all plots) | Use Plotly for all plots if interactive exploration is the primary goal. However, use Matplotlib for publication-quality static figures. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Manual experiment scripts** | No systematic tracking, hard to compare results, easy to lose what worked | Hydra + MLflow for configuration and tracking |
| **Jupyter notebooks for experiments** | Version control issues, hidden state, hard to run many experiments systematically | Python scripts with Hydra configs; use notebooks only for exploratory analysis |
| **Grid search** | Extremely inefficient; searches irrelevant hyperparameter space | Optuna with pruning (Bayesian optimization) |
| **Global configuration files** | Cannot compose configs; duplicate parameters across experiments | Hydra hierarchical configs (model + data + trainer) |
| **Custom tracking (CSV/JSON)** | No UI, hard to compare, missing metadata (git hash, timestamps) | MLflow tracking with auto-logging |
| **Sequential experiment execution** | Wastes GPU/CPU resources; slow iteration | Ray Tune for parallel execution across devices |
| **Single random seed** | Results may be luck; need robustness checks | 5-fold CV with multiple seeds; report mean ± std |
| **Training on full data for stacking** | Data leakage causes optimistic bias | OOF (out-of-fold) predictions for meta-learner training |
| **NDVI alone for dead biomass** | NDVI cannot detect dead matter; systematic under-prediction | K-Means color segmentation + spectral indices (VARI, GLI, NGRDI) |

## Stack Patterns by Variant

**If running systematic ablations (model components, features):**
- Use Hydra config inheritance to define base config + ablation overrides
- Use MLflow to log all ablations as child runs under a parent experiment
- Use Pandas to aggregate results and compare ablations in a table

**If optimizing hyperparameters:**
- Use Optuna with pruning (stop bad trials early)
- Use Ray Tune to parallelize trials across CPU cores
- Use MLflow to log each trial as a separate run with parameters and metrics

**If comparing ensemble strategies:**
- Use Hydra multirun to define ensemble configs (Ridge, Quantile, Stacking)
- Use OOF predictions to avoid data leakage
- Use Seaborn heatmaps to visualize correlation between base models

**If analyzing model errors:**
- Use SHAP to identify which features cause high-error predictions
- Use Matplotlib/Seaborn to plot residuals vs predicted values, actual values
- Use Pandas to group errors by species, state, or biomass range

**If running post-hoc analysis on completed experiments:**
- Use MLflow UI to load experiment results
- Use Pandas to read logged metrics and parameters
- Use Plotly for interactive exploration of experiment space

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch 2.5+ | PyTorch Lightning 2.4+ | Lightning 2.4 requires PyTorch 2.0+; use matching versions |
| Ray 2.49+ | Python 3.8-3.12 | Ray dropped Python 3.7 support; use Python 3.10+ for best compatibility |
| MLflow 2.17+ | PyTorch 2.0+, scikit-learn 1.0+ | MLflow autologging supports recent versions; test integrations before relying on them |
| Optuna 4.5+ | PyTorch Lightning 2.0+ | Optuna integration with Lightning requires recent versions |
| Hydra 1.3+ | Python 3.6-3.11 | Hydra 1.3 stable; Hydra 1.4 (beta) adds Python 3.12 support |
| XGBoost 2.1+ | scikit-learn 1.0+ | XGBoost sklearn API requires scikit-learn |
| SHAP 0.46+ | XGBoost 1.0+, PyTorch 1.0+ | SHAP supports tree models and deep models; install both backends |

## Experiment Framework Architecture

Based on research, recommended structure for systematic experimentation:

```
experiments/
├── configs/                    # Hydra configs
│   ├── model/                 # Model architectures
│   │   ├── resnet18.yaml
│   │   ├── efficientnet_b0.yaml
│   │   └── xgboost_tabular.yaml
│   ├── data/                  # Data configurations
│   │   ├── default.yaml
│   │   └── augmentation.yaml
│   ├── trainer/               # Training configurations
│   │   ├── default.yaml
│   │   └── tta.yaml
│   └── experiment/            # Experiment configs (compose model + data + trainer)
│       ├── baseline_tabular.yaml
│       ├── image_cnn.yaml
│       └── stacking_ensemble.yaml
├── scripts/                    # Experiment scripts
│   ├── run_experiment.py      # Main entry point (Hydra + MLflow)
│   ├── optimize_hyperparams.py # Optuna + Ray Tune
│   └── analyze_results.py     # Load MLflow runs, aggregate with Pandas
└── results/                    # Auto-generated by MLflow
    └── mlruns/
```

**Key patterns:**
1. **Hydra for composition:** Each experiment config composes model + data + trainer configs
2. **MLflow for tracking:** Every script call logs to MLflow with parameters, metrics, artifacts
3. **Optuna for search:** Hyperparameter optimization scripts log each trial to MLflow
4. **Pandas for analysis:** Post-hoc scripts load MLflow results into DataFrames for comparison

## Sources

### Official Documentation (HIGH Confidence)
- **MLflow** — https://mlflow.org/docs/latest/index.html — Verified v2.17.0rc0, tracking, models, registry
- **Optuna** — https://optuna.readthedocs.io/en/stable/ — Verified v4.5.0, pruning, define-by-run API, visualization
- **Ray Tune** — https://docs.ray.io/en/latest/tune/index.html — Verified v2.49.2, scalable tuning, fault tolerance
- **Hydra** — https://hydra.cc/docs/intro — Verified v1.3 stable, hierarchical configs, multirun mode
- **Weights & Biases** — https://docs.wandb.ai/guides — Verified experiment tracking, sweeps, visualization
- **PyTorch Lightning** — https://lightning.ai/docs/pytorch-lightning/stable/ — Verified training framework, boilerplate reduction

### Research Context (MEDIUM Confidence)
- Project codebase analysis — Existing experiments use PyTorch, XGBoost, scikit-learn
- PROJECT.md — Documented 13 experiments, current RMSE ~11.4, known issues with high-biomass bias

### Verified Patterns (HIGH Confidence)
- OOF predictions for stacking — Standard methodology to prevent data leakage
- 5-fold CV over single-split — Single-split (6.64) optimistic; 5-fold (11.34) more reliable
- K-Means for dead matter — NDVI cannot detect non-green biomass; clustering helps
- Grid features over global — Spatial heterogeneity requires local indices

### Community Best Practices (MEDIUM Confidence)
- SHAP for explainability — Already used successfully in Phase 10
- Parallel execution with Ray — Standard for scaling ML experiments
- Hydra composition — Industry standard for ML research (Facebook, NVIDIA, etc.)

---
*Stack research for: ML Experimentation Framework for Biomass Prediction*
*Researched: 2025-01-17*
