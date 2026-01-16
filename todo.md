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
- [ ] **Tabular Baseline**
    - [ ] Prepare standard X/y splits (Train/Val).
    - [ ] Train XGBoost/CatBoost using only `Height`, `NDVI`, `State`, `Species`.
    - [ ] Evaluate RMSE/MAE.
- [ ] **Simple CNN Baseline**
    - [ ] Create PyTorch/TF Dataset class.
    - [ ] Train a ResNet18 predicting `Dry_Total_g` from images only.

## Phase 3: Core Development (Multimodal)
- [ ] **Data Loader**
    - [ ] detailed implementation handling both Image and Tabular data.
    - [ ] Augmentations (Flip, Rotate, ColorJitter).
- [ ] **Model Architecture**
    - [ ] Implement `BiomassPredictor` class.
    - [ ] Image Encoder: EfficientNet or ResNet.
    - [ ] Tabular Encoder: MLP.
    - [ ] Fusion Layer: Concat -> Dense -> Output(5).
- [ ] **Training Loop**
    - [ ] Loss function (MSE or smooth L1).
    - [ ] Optimizer (AdamW).
    - [ ] Schedulers (CosineAnnealing).

## Phase 4: Refinement & Submission
- [ ] **Evaluation**
    - [ ] Compare validation metrics across experiments.
    - [ ] Error analysis (where does it fail?).
- [ ] **Inference**
    - [ ] Script to generate `sample_submission.csv`.
    - [ ] Final submission generation.
