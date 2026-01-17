# GEMINI Project Log

## 2026-01-16
- **Phase 3 Complete**: Implemented Multimodal (ResNet18 + MLP) model.
- **Results**: Multimodal RMSE (14.33) is better than Image-only (28.6) but worse than Tabular Baseline (10.92).
- **Next Steps**: Focus on Tabular Refinement (Phase 4).

- **Phase 4 Complete**: Refined XGBoost models using RandomizedSearchCV.
- **Results**: Optimized Tabular RMSE (11.60) confirms Metadata (Height/NDVI) is the primary signal.
- **Submission**: Generated `submission.csv` using Training Mean/Mode imputation for missing metadata in test set.

- **Phase 6 Complete**: Segmentation-Augmented Ensembling.
- **Key Breakthrough**: **Experiment 6 (K-Means Segmentation)**.
    - Used unsupervised clustering to extract Color/Fraction features for Soil, Dead, and Green partitions.
    - Achieved RMSE 13.45 (Single model), dramatically better than simple color features (17.95).
- **Final Result**: **Experiment 7 (Ensemble)**.
    - 3-way blend: Metadata Proxy (Tabular) + EfficientNet (Deep Learning) + K-Means (Segmentation).
    - **Outcome**: Breakthrough **Validation RMSE: 6.64**.
    - **Conclusion**: K-Means features are the "missing link" for quantifying dry/dead matter, which spectral (NDVI) and standard CNNs struggle with on small datasets.
- **Submission**: `submission_ensemble.csv` generated.
- **Status**: Research and Development Complete.
