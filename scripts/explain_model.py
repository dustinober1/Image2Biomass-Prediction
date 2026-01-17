
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import json

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
MODEL_DIR = 'models/exp13_quantile'
OUTPUT_DIR = 'models/explainability'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature files
OOF_TABULAR = 'models/stacking/tabular/oof_tabular.csv'
OOF_KMEANS = 'models/stacking/kmeans/oof_kmeans.csv'
OOF_EFFNET = 'models/stacking/effnet/oof_effnet.csv'
FEATURES_GRID = 'models/features_grid/features_grid_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def load_all_features():
    df_train = pd.read_csv(TRAIN_CSV)
    wide_df = df_train.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    oof_tab = pd.read_csv(OOF_TABULAR)
    oof_km = pd.read_csv(OOF_KMEANS)
    oof_eff = pd.read_csv(OOF_EFFNET)
    grid_df = pd.read_csv(FEATURES_GRID)
    
    stacked_df = pd.merge(wide_df, oof_tab, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_km, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_eff, on='image_path')
    stacked_df = pd.merge(stacked_df, grid_df, on='image_path')
    
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
    meta_df = df_train.groupby('image_path')[meta_cols].first().reset_index()
    stacked_df = pd.merge(stacked_df, meta_df, on='image_path')
    
    drop_cols = ['image_path'] + TARGETS
    feature_cols = [c for c in stacked_df.columns if c not in drop_cols]
    
    return stacked_df[feature_cols], feature_cols

def explain_model():
    X, feature_names = load_all_features()
    print(f"Loaded {len(feature_names)} features for explainability.")
    
    for target in TARGETS:
        print(f"\nProcessing SHAP for {target}...")
        model_path = os.path.join(MODEL_DIR, f'lgbm_meta_{target}.pkl')
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        model = joblib.load(model_path)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plot Summary (Top 10 features)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=15, show=False)
        plt.title(f"SHAP Feature Importance: {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_summary_{target}.png'))
        plt.close()
        
        # Plot Interaction for top feature
        # (Identifying top feature from mean absolute shap values)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_feat_idx = np.argmax(mean_abs_shap)
        top_feat_name = feature_names[top_feat_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feat_name, shap_values, X, show=False)
        plt.title(f"SHAP Dependence: {top_feat_name} ({target})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_dep_{target}_{top_feat_name}.png'))
        plt.close()

    print(f"\nExplainability reports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    explain_model()
