# Project Plan: Image2Biomass Prediction

## Goal
Predict 5 pasture biomass components using top-view images and auxiliary field data (NDVI, Height, Location, Species).

## Data Overview
- **Input Content**:
    - **Images**: Top-view pasture images (`train/`, `test/`).
    - **Metadata**: `Height_Ave_cm`, `Pre_GSHH_NDVI` (GreenSeeker), `State`, `Species`, `Sampling_Date`.
- **Targets**:
    1. `Dry_Green_g`
    2. `Dry_Dead_g`
    3. `Dry_Clover_g`
    4. `GDM_g` (Green Dry Matter)
    5. `Dry_Total_g`
- **Challenge**: The test set requests specific target components for specific images.

## Strategy

### 1. Data Exploration & Analysis (EDA)
- **Objective**: Understand data distributions, correlations, and quality.
- **Key Questions**:
    - How correlated are `NDVI` and `Height` with the biomass targets?
    - Are the 5 targets correlated with each other? (e.g. `Dry_Total_g` should roughly sum others).
    - Class imbalance?
    - Image resolution and quality checks.

### 2. Preprocessing
- **Images**: Resize/Crop (e.g., 224x224 or 512x512). Normalize using ImageNet stats.
- **Metadata**:
    - One-hot encode `State` and `Species`.
    - Normalize/Scale `Height` and `NDVI`.
    - Date features? (Seasonality might happen).

### 3. Model Architecture
We will approach this as a **Multimodal Regression** problem.

**A. Tabular Branch (Auxiliary Data)**
- Inputs: `Height`, `NDVI`, `State` (embeddings), `Species` (embeddings).
- Model: MLP (Multi-Layer Perceptron) or Gradient Boosting (for baseline).

**B. Vision Branch (Images)**
- Inputs: Raw RGB Images.
- Model: CNN Backbone (EfficientNetV2, ConvNeXt, or ResNet).
- Pretrained on ImageNet.

**C. Fusion**
- Concatenate feature vectors from Vision Head and Tabular Head.
- Pass through final regression layers to output 5 values.
- *Note:* Even if test asks for one specific target, predicting all 5 jointly helps the model learn relationships (Multi-Task Learning).

### 4. Experiments & Hypothesis
| ID | Model Type | Inputs | Hypothesis |
|----|------------|--------|------------|
| **Exp 1** | Tabular Baseline | Height, NDVI, Meta | `Height` and `NDVI` are strong predictors; this sets the "floor" performance. |
| **Exp 2** | Vision Only | Images | Images contain texture/color density info not captured by simple height/NDVI. |
| **Exp 3** | **Multimodal (Fusion)** | Images + Meta | **(Primary Hypothesis)** Combining both yields best results. Height gives volume proxy; Image gives density/composition. |
| **Exp 4** | Loss Function | - | MSE vs MAE. |
| **Exp 5** | Multi-Task vs Single | - | Predicting all 5 simultaneously regularizes the model better than training 5 separate models. |

## Timeline & Milestones
1.  **Exploration**: stats, plots, data loader setup.
2.  **Baseline**: Get a score with just xgboost on csv.
3.  **Deep Learning Setup**: PyTorch dataset/dataloader, simple ResNet.
4.  **Fusion**: Combine inputs.
5.  **Refinement**: Hyperparameter tuning, larger models.
