# Project Todo List

## Phase 1: Setup & Data Exploration
- [x] **Data Inspection**
    - [x] Load `train.csv` and inspect headers/types.
    - [x] Check for missing values.
    - [x] Visualize distribution of `target` variables.
    - [x] Plot correlation matrix (Targets vs NDVI vs Height).
- [x] **Image Inspection**
    - [x] Load and display random 5-10 images.
    - [x] Check image dimensions and aspect ratios.

## Phase 2: Baselines
- [x] **Tabular Baseline**
    - [x] Prepare standard X/y splits (Train/Val).
    - [x] Train XGBoost on `Height`, `NDVI`, `State`, `Species`.
    - [x] Evaluate RMSE/MAE. (RMSE ~10.9)
- [x] **Simple CNN Baseline**
    - [x] Create PyTorch Dataset class.
    - [x] Train a ResNet18 predicting `Dry_Total_g` from images only. (RMSE ~28.6)

## Phase 3: Core Development (Multimodal)
- [x] **Data Loader**
    - [x] detailed implementation handling both Image and Tabular data.
    - [x] Augmentations (Flip, Rotate, ColorJitter).
- [x] **Model Architecture**
    - [x] Implement `BiomassPredictor` class.
    - [x] Image Encoder: EfficientNet or ResNet.
    - [x] Tabular Encoder: MLP.
    - [x] Fusion Layer: Concat -> Dense -> Output(5).
- [x] **Training Loop**
    - [x] Loss function (MSE or smooth L1).
    - [x] Optimizer (AdamW).
    - [x] Schedulers (CosineAnnealing).

## Phase 4: Refinement & Submission
- [x] **Evaluation**
    - [x] Compare validation metrics across experiments.
    - [x] Error analysis (where does it fail?).
- [x] **Inference**
    - [x] Script to generate `sample_submission.csv`.
    - [x] Final submission generation.
