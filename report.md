# Data Exploration Report

## Dataset Overview
- **Total Images**: 357 (Training Set)
- **Observations**: 1785 (5 targets per image)
- **Image Dimensions**: 2000x1000 pixels (Landscape)
- **Image Format**: RGB JPEG

## Data Quality
- **Missing Values**: None found in training metadata or targets.
- **Class Imbalance**:
    - `Dry_Clover_g` has a very low mean and high variance, indicating it might be sparse (many zeros or low values).
    - `Dry_Dead_g` also varies significantly.

## Correlations
| Feature | Dry_Green_g | Dry_Dead_g | Dry_Clover_g | GDM_g | Dry_Total_g |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Height_Ave_cm** | 0.69 | 0.20 | 0.15 | 0.69 | 0.67 |
| **Pre_GSHH_NDVI** | 0.75 | -0.19 | 0.16 | 0.76 | 0.63 |

**Key Insights**:
1.  **Strong Predictors**: `NDVI` and `Height` are strong predictors for Green Biomass (`Dry_Green_g`, `GDM_g`) and Total Biomass (`Dry_Total_g`).
2.  **Weak Predictors for Dead/Clover**: They correlate poorly with `Dry_Dead_g` and `Dry_Clover_g`. The image model will be crucial here as spectral data (NDVI) might not capture dead matter well (which isn't green).
3.  **Redundancy**: `GDM_g` (Green Dry Matter) and `Dry_Green_g` are extremely highly correlated (likely measuring similar things), suggesting multi-task learning will share significant features.

## Image Properties
- **Resolution**: High resolution (2000x1000).
- **Strategy**:
    - Resizing to standard 224x224 might lose too much detail given the aspect ratio (2:1).
    - **Recommendation**: Resize to 512x256 or use random crops of 512x512 during training to maintain texture details.

## Next Steps (Phase 2)
1.  **Baseline Model**: Train a Gradient Boosting model (XGBoost/CatBoost) using only `Height`, `NDVI`, `Species`, `State` to establish a performance floor.
2.  **Image Pipeline**: Build a PyTorch Dataset that loads images and resizes/pads them correctly.

## Phase 2: Baseline Modeling Results

### 1. Tabular Baseline (XGBoost)
- **Inputs**: `Height_Ave_cm`, `Pre_GSHH_NDVI`, `State`, `Species`
- **Performance**:
    - **Avg RMSE**: **10.92**
    - **Avg R2**: **0.62**
- **Detailed Findings**:
    - **Green Biomass**: Highly predictable using `NDVI` and `Height` (R2 ~0.78 for `Dry_Green_g`).
    - **Dead Biomass**: Poorly predicted (R2 ~0.37 for `Dry_Dead_g`), as NDVI primarily measures greenness.
    - **Implication**: Metadata sets a strong floor for green components but fails on dead/dry matter.

### 2. Image Baseline (ResNet18)
- **Inputs**: Raw RGB Images (resized to 224x224)
- **Performance**:
    - **Avg RMSE**: **~28.6** (Best Validation RMSE)
    - **Avg R2**: Negative or near zero (Poor generalization)
- **Detailed Findings**:
    - The model struggled significantly compared to the tabular baseline.
    - **Why?**: Biomass density is hard to estimate purely from 2D texture without depth (height) or spectral calibration (NDVI) information in this small dataset (357 images).
    - **Implication**: Visual features alone are **insufficient**.

## Phase 3: Multimodal Modeling Results

### 1. Multimodal Early Fusion (ResNet18 + MLP)
- **Architecture**: ResNet18 (Image) + 3-Layer MLP (Tabular) -> Concat -> Fusion Head.
- **Training**: 100 Epochs, AdamW, CosineAnnealing, SmoothL1Loss.
- **Augmentations**: RandomCrop, Flip, Rotation, ColorJitter.
- **Performance**:
    - **Avg RMSE**: **14.33**
    - **Avg R2**: **0.28**
- **Target Breakdown**:
    - **Best**: `GDM_g` (R2=0.67), `Dry_Green_g` (R2=0.54).
    - **Poor**: `Dry_Dead_g` (R2=-0.19), `Dry_Clover_g` (R2=-0.10).
- **Comparison**:
    - **Vs Tabular**: Multimodal (**14.33 RMSE**) performs **worse** than Tabular Baseline (**10.92 RMSE**).
    - **Vs Image**: Significantly better than Image-only (**28.6 RMSE**).
- **Conclusion**:
    - Adding images currently **degrades** performance compared to pure metadata.
    - The small dataset size (357 images) likely causes the CNN branch to overfit or learn noise, confusing the fusion layer.
    - **Recommendation for Phase 4**: Focus on the Tabular model for final submission, or try to regularize the image branch heavily (freeze weights, reduce complexity).

## Conclusion & Next Steps (Phase 4)
- **Strategy**: The Metadata signal (`Height`, `NDVI`) is the most reliable.
- **Hypothesis**: Complex deep learning is overfitting. Ensemble tree-based methods (XGBoost/CatBoost) on metadata will likely yield the best leaderboard score.
- **Plan**:
    1.  **Refinement**: Optimize the XGBoost/CatBoost models.
    2.  **Submission**: Generate final predictions using the Tabular pipeline.
