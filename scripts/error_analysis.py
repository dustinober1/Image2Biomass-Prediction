
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/error_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature files
OOF_TABULAR = 'models/stacking/tabular/oof_tabular.csv'
OOF_KMEANS = 'models/stacking/kmeans/oof_kmeans.csv'
OOF_EFFNET = 'models/stacking/effnet/oof_effnet.csv'
FEATURES_GRID = 'models/features_grid/features_grid_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def run_error_analysis():
    print("Running Error Analysis...")
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
    
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'Species', 'State']
    meta_df = df_train.groupby('image_path')[meta_cols].first().reset_index()
    stacked_df = pd.merge(stacked_df, meta_df, on='image_path')
    
    drop_cols = ['image_path'] + TARGETS + ['Species', 'State']
    feature_cols = [c for c in stacked_df.columns if c not in drop_cols]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof_preds = np.zeros((len(stacked_df), len(TARGETS)))
    
    # We re-run the meta-learner training to get OOF mapped to image_path
    for target_idx, target in enumerate(TARGETS):
        print(f"  Calculating OOF for {target}...")
        y = stacked_df[target].values
        X = stacked_df[feature_cols].values
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=31,
                verbose=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])
            
            meta_oof_preds[val_idx, target_idx] = model.predict(X_val)
            
    # Calculate Residuals
    for i, target in enumerate(TARGETS):
        error_col = f'Error_{target}'
        stacked_df[f'Pred_{target}'] = meta_oof_preds[:, i]
        stacked_df[error_col] = np.abs(stacked_df[target] - stacked_df[f'Pred_{target}'])
        
    # Calculate Mean Absolute Error across all targets per image
    error_cols = [f'Error_{t}' for t in TARGETS]
    stacked_df['Avg_MAE'] = stacked_df[error_cols].mean(axis=1)
    
    # Find top 10 hardest images
    hardest_df = stacked_df.sort_values(by='Avg_MAE', ascending=False).head(10)
    
    cols_to_show = ['image_path', 'Avg_MAE', 'Height_Ave_cm', 'Pre_GSHH_NDVI', 'Species', 'State'] + TARGETS + [f'Pred_{t}' for t in TARGETS]
    hardest_df[cols_to_show].to_csv(os.path.join(OUTPUT_DIR, 'hardest_samples.csv'), index=False)
    
    print(f"\nTop 5 Hardest Samples:")
    for _, row in hardest_df.head(5).iterrows():
        print(f"Image: {row['image_path']}, MAE: {row['Avg_MAE']:.2f}, Species: {row['Species']}")

    # Summary by Species
    species_err = stacked_df.groupby('Species')['Avg_MAE'].mean().sort_values(ascending=False)
    print("\nMean Error by Species:")
    print(species_err)
    
    stacked_df.to_csv(os.path.join(OUTPUT_DIR, 'full_error_log.csv'), index=False)
    print(f"\nError analysis saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_error_analysis()
