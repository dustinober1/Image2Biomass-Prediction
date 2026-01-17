
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/stacking/kmeans'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FEATS_KMEANS_TRAIN = 'models/features_kmeans/features_kmeans_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_oof():
    df = pd.read_csv(TRAIN_CSV)
    
    # Pivot Targets
    wide_df = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Metadata for base features
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
    meta_df = df.groupby('image_path')[meta_cols].first().reset_index()
    wide_df = pd.merge(wide_df, meta_df, on='image_path')
    
    # KMeans Features
    km_df = pd.read_csv(FEATS_KMEANS_TRAIN)
    wide_df = pd.merge(wide_df, km_df, on='image_path')
    
    km_feats = [c for c in wide_df.columns if 'KM_' in c]
    FEATURES = ['Height_Ave_cm', 'Pre_GSHH_NDVI'] + km_feats
    
    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(wide_df), len(TARGETS)))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(wide_df)):
        print(f"\n--- Fold {fold} ---")
        train_fold = wide_df.iloc[train_idx]
        val_fold = wide_df.iloc[val_idx]
        
        for i, target in enumerate(TARGETS):
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(
                train_fold[FEATURES], 
                train_fold[target],
                eval_set=[(val_fold[FEATURES], val_fold[target])],
                verbose=False
            )
            
            # Save Model
            model.save_model(os.path.join(OUTPUT_DIR, f'xgb_{target}_fold{fold}.json'))
            
            # OOF
            oof_preds[val_idx, i] = model.predict(val_fold[FEATURES])
            
    # Save OOF
    oof_df = pd.DataFrame(oof_preds, columns=[f'OOF_KMeans_{t}' for t in TARGETS])
    oof_df['image_path'] = wide_df['image_path']
    oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_kmeans.csv'), index=False)
    
    # Score
    total_rmse = np.sqrt(mean_squared_error(wide_df[TARGETS], oof_preds))
    print(f"\nOverall KMeans OOF RMSE: {total_rmse:.4f}")

if __name__ == "__main__":
    train_oof()
