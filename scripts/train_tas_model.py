
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import json

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = 'train.csv'
FEATURES_CSV = 'models/features_kmeans/features_kmeans_train.csv'
OUTPUT_DIR = 'models/exp9_tas'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_exp9():
    print("Loading data for Experiment 9 (TAS)...")
    df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV))
    
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
    
    # Pivot Targets
    wide_df = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Extract Metadata
    if set(meta_cols).issubset(df.columns):
        meta_df = df.groupby('image_path')[meta_cols].first().reset_index()
        wide_df = pd.merge(wide_df, meta_df, on='image_path')
    
    # Merge with TAS Features
    feat_df = pd.read_csv(FEATURES_CSV)
    full_df = pd.merge(wide_df, feat_df, on='image_path')
    
    print(f"Data Loaded. Shape: {full_df.shape}")
    
    # Features to use: Metadata + TAS (KMeans + Std Dev)
    drop_cols = ['image_path'] + TARGETS
    feature_cols = [c for c in full_df.columns if c not in drop_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overall_rmse = []
    results = {}
    
    for target in TARGETS:
        print(f"\nTraining TAS for {target}...")
        y = full_df[target].values
        X = full_df[feature_cols].values
        
        target_rmses = []
        target_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            target_rmses.append(rmse)
            target_r2s.append(r2)
        
        avg_rmse = np.mean(target_rmses)
        avg_r2 = np.mean(target_r2s)
        print(f"  -> Avg RMSE: {avg_rmse:.4f}, Avg R2: {avg_r2:.4f}")
        overall_rmse.append(avg_rmse)
        results[target] = {'rmse': avg_rmse, 'r2': avg_r2}
        
        # Train Final Model on All Data
        final_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X, y)
        final_model.save_model(os.path.join(OUTPUT_DIR, f'xgb_tas_{target}.json'))
        
    mean_rmse = np.mean(overall_rmse)
    print(f"\nOverall TAS Validation RMSE: {mean_rmse:.4f}")
    
    # Save results
    results['overall_rmse'] = mean_rmse
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_exp9()
