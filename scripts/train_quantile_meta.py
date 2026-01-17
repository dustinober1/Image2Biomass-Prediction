
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import json

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/exp13_quantile'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OOF Files
OOF_TABULAR = 'models/stacking/tabular/oof_tabular.csv'
OOF_KMEANS = 'models/stacking/kmeans/oof_kmeans.csv'
OOF_EFFNET = 'models/stacking/effnet/oof_effnet.csv'

# New Features
FEATURES_GRID = 'models/features_grid/features_grid_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_quantile_meta():
    print("Loading data for Experiment 13 (Quantile Stacking)...")
    df_train = pd.read_csv(TRAIN_CSV)
    wide_df = df_train.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Load OOFs
    oof_tab = pd.read_csv(OOF_TABULAR)
    oof_km = pd.read_csv(OOF_KMEANS)
    oof_eff = pd.read_csv(OOF_EFFNET)
    
    # Merge OOFs
    stacked_df = pd.merge(wide_df, oof_tab, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_km, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_eff, on='image_path')
    
    # Load Grid Features
    grid_df = pd.read_csv(FEATURES_GRID)
    stacked_df = pd.merge(stacked_df, grid_df, on='image_path')
    
    # Also get raw metadata
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
    meta_df = df_train.groupby('image_path')[meta_cols].first().reset_index()
    stacked_df = pd.merge(stacked_df, meta_df, on='image_path')
    
    print(f"Dataset shape: {stacked_df.shape}")
    
    # Features Selection
    drop_cols = ['image_path'] + TARGETS
    feature_cols = [c for c in stacked_df.columns if c not in drop_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overall_rmse = []
    results = {}
    
    for target in TARGETS:
        print(f"\nTraining Meta-Learner for {target}...")
        y = stacked_df[target].values
        X = stacked_df[feature_cols].values
        
        target_rmses = []
        target_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Robust Quantile Stacking: Use Quantile Loss (Median regression)
            # This is less sensitive to outliers in base models
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5, # Median
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            target_rmses.append(rmse)
            target_r2s.append(r2)
            
        avg_rmse = np.mean(target_rmses)
        avg_r2 = np.mean(target_r2s)
        print(f"  -> Quantile OOF RMSE: {avg_rmse:.4f}, R2: {avg_r2:.4f}")
        overall_rmse.append(avg_rmse)
        results[target] = {'rmse': float(avg_rmse), 'r2': float(avg_r2)}
        
        # Train Final Meta-Model
        final_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.5,
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42
        )
        final_model.fit(X, y)
        import joblib
        joblib.dump(final_model, os.path.join(OUTPUT_DIR, f'lgbm_meta_{target}.pkl'))
        
    final_avg_rmse = np.mean(overall_rmse)
    print(f"\nOverall Quantile Stacking RMSE: {final_avg_rmse:.4f}")
    results['overall_rmse'] = float(final_avg_rmse)
    
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_quantile_meta()
