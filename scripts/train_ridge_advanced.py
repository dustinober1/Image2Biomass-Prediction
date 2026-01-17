
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import json

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/exp14_ridge_advanced'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OOF_TABULAR = 'models/stacking/tabular/oof_tabular.csv'
OOF_KMEANS = 'models/stacking/kmeans/oof_kmeans.csv'
OOF_EFFNET = 'models/stacking/effnet/oof_effnet.csv'
FEATURES_GRID = 'models/features_grid/features_grid_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_ridge_advanced():
    print("Training Ridge Meta-Learner with Advanced Features...")
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
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    overall_rmse = []
    
    for target in TARGETS:
        y = stacked_df[target].values
        X = stacked_df[feature_cols].values
        
        target_rmses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            target_rmses.append(rmse)
            
        avg_rmse = np.mean(target_rmses)
        print(f"  -> {target} Ridge RMSE: {avg_rmse:.4f}")
        overall_rmse.append(avg_rmse)
        
    print(f"\nOverall Ridge + Advanced Feats RMSE: {np.mean(overall_rmse):.4f}")

if __name__ == "__main__":
    train_ridge_advanced()
