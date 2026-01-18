# Image2Biomass Experimental Framework

## What This Is

A systematic experimental research framework for pasture biomass prediction. This project has completed 13 experiments across 10 phases, establishing a hierarchical stacking ensemble (RMSE ~11.4) as the current best approach. The next phase focuses on building a robust experimentation platform to run controlled ablations, comparisons, and analyses to understand why models work and discover further improvements.

## Core Value

**Understand what drives biomass predictions.** Not just "what works" but "why it works" — enabling systematic discovery of better models through controlled experimentation rather than trial-and-error.

## Requirements

### Validated

- ✓ **Biomass prediction models** — 13 experimental approaches validated (RMSE 6.64 single-split, 11.34 5-fold CV) — existing
- ✓ **Tabular baseline** — XGBoost on Height/NDVI achieves RMSE ~11 — existing
- ✓ **Image models** — ResNet18, EfficientNet-B0 with TTA — existing
- ✓ **Multimodal fusion** — Combines images with metadata — existing
- ✓ **Ensemble strategies** — Hierarchical stacking, weighted blends, quantile stacking — existing
- ✓ **Feature extraction** — K-Means segmentation, grid spectral indices (VARI, GLI, NGRDI) — existing
- ✓ **Explainability** — SHAP analysis, error analysis completed — existing

### Active

- [ ] **Model ablations** — Controlled comparisons of model architectures and components
- [ ] **Feature ablations** — Test with/without specific features to measure contribution
- [ ] **Ensemble comparisons** — Compare stacking strategies (Ridge vs Quantile vs non-linear meta-learners)
- [ ] **Insights documentation** — Structured findings on what works, what doesn't, and why

### Validated

- ✓ **Experimental framework** — v1 shipped with MLflow tracking, YAML configs, batch execution, hyperparameter optimization, and advanced analytics — 2026-01-18

### Out of Scope

- **Data collection** — Dataset is fixed (357 training images, 5 targets per image)
- **Real-time inference** — Focus is on research, not production deployment
- **Mobile/web app** — No UI requirements; this is a research project
- **New data sources** — Working with existing images, metadata, and NDVI values only

## Context

**Dataset:**
- 357 training images (2000x1000 RGB)
- 5 biomass targets per image: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g
- Metadata: Height_Ave_cm, Pre_GSHH_NDVI, State, Species
- High correlations: Height/Green (0.69), NDVI/Green (0.75)

**Key Findings from 13 Experiments:**
1. **Metadata dominates** — Height and NDVI are the strongest predictors (tabular RMSE ~11)
2. **Images need care** — Raw CNNs overfit on N=357; needs transfer learning, TTA, or careful architecture
3. **Dead matter problem** — NDVI cannot detect dead biomass; segmentation (K-Means) helps
4. **Spatial information** — Grid-based spectral indices outperform global averages
5. **Ensemble wins** — Stacking diverse models achieves best robustness (RMSE 11.34 5-fold)

**Current Best:**
- Hierarchical stacking ensemble
- Base models: Tabular metadata, EfficientNet-B0, K-Means segmentation
- Meta-learner: Ridge/Quantile stacking
- OOF RMSE: ~11.34 (5-fold CV), 6.64 (single-split, optimistic)

**Known Issues:**
- Conservative bias on high-biomass samples (>120g) — under-predicts
- Species variance: Fescue and Lucerne show higher error (density saturation)
- Small dataset limits deep learning generalization
- Test set lacks metadata; requires proxy models or imputation

## Current State

**Shipped:** v1 (Experiment Tracking Foundation) — 2026-01-18

**Delivered:**
- MLflow-based experiment tracking with reproducibility guarantees
- YAML-driven configuration with parameter sweeps
- Batch execution with resource management
- Hyperparameter optimization with Optuna
- Advanced analytics (error analysis, interpretability, insights)
- Complete end-to-end workflows (training → artifacts → analytics)

**Codebase:**
- ~17,172 lines of Python code
- 12 phases, 21 plans completed
- 27/27 requirements satisfied (24 v1 + 3 v2 early)
- Zero tech debt

**Next Milestone Goals:**
- Model ablations (controlled architecture comparisons)
- Feature ablations (with/without specific features)
- Ensemble comparisons (stacking strategies)
- Insights documentation (structured findings)

## Constraints

- **Dataset size** — Fixed at 357 training images; cannot collect more
- **Compute** — GPU available (used for CNN training); experiments can run in parallel
- **Time** — Batch experimentation preferred (define many, run, analyze results)
- **Reproducibility** — Need systematic tracking of experiments, not ad-hoc scripts

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 5-Fold CV over single-split | Single-split (6.64) optimistic; 5-fold (11.34) more reliable | ✓ Good — Robust estimate |
| K-Means for dead matter | NDVI blind to non-green; clustering segments by color | ✓ Good — Improved dead biomass |
| Grid features over global | Spatial heterogeneity in pasture; local indices more informative | ✓ Good — 81+ features useful |
| Hierarchical stacking | Simple average ignores model specializations; meta-learner learns weights | ✓ Good — Best overall performance |
| OOF predictions for stacking | Prevents data leakage; ensures meta-learner trains on unbiased predictions | ✓ Good — Proper CV methodology |

---
*Last updated: 2026-01-18 after v1 milestone*
