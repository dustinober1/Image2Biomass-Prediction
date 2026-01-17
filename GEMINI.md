# GEMINI Project Log

## 2026-01-16
- **Phase 3 Complete**: Implemented Multimodal (ResNet18 + MLP) model.
- **Results**: Multimodal RMSE (14.33) is better than Image-only (28.6) but worse than Tabular Baseline (10.92).
- **Next Steps**: Focus on Tabular Refinement (Phase 4).

- **Phase 4 Complete**: Refined XGBoost models using RandomizedSearchCV.
- **Results**: Optimized Tabular RMSE (11.60) confirms Metadata (Height/NDVI) is the primary signal.
- **Submission**: Generated `submission.csv` using Training Mean/Mode imputation for missing metadata in test set.

- **Phase 5 Complete**: Advanced Experiments (Test Set Adaptation).
- **Strategy**: Tested 5 strategies to handle missing metadata in test images.
- **Key Success**: **Experiment 1 (Metadata Proxy)**.
    - Trained CNN to predict `Height` from images (R2=0.87).
    - Used predicted height to feed the Tabular Model.
    - **Outcome**: `submission_exp1.csv` is the most scientifically robust submission, leveraging the strong tabular performance.
- **Secondary Success**: **Experiment 2 (Visual Features)**.
    - Simple features (RGB/Texture) achieved RMSE 17.9, significantly beating Deep Learning baselines (RMSE 28.6), proving "less is more" for small data.
- **Status**: Project Implementation Complete.
