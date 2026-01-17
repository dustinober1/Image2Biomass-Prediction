# GEMINI Project Log

## 2026-01-16
- **Phase 3 Complete**: Implemented Multimodal (ResNet18 + MLP) model.
- **Results**: Multimodal RMSE (14.33) is better than Image-only (28.6) but worse than Tabular Baseline (10.92).
- **Next Steps**: Focus on Tabular Refinement (Phase 4).

- **Phase 4 Complete**: Refined XGBoost models using RandomizedSearchCV.
- **Results**: Optimized Tabular RMSE (11.60) confirms Metadata (Height/NDVI) is the primary signal.
- **Submission**: Generated `submission.csv` using Training Mean/Mode imputation for missing metadata in test set.
- **Status**: Project Implementation Complete.
