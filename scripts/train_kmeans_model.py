
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
TEST_FEATURES_CSV = 'models/features_kmeans/features_kmeans_test.csv'
OUTPUT_DIR = 'models/exp6_kmeans'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def load_data():
    # Load targets
    df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV))
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Load features
    features_df = pd.read_csv(FEATURES_CSV)
    
    # Merge
    merged_df = pd.merge(targets_wide, features_df, on='image_path', how='inner')
    
    # We also need metadata (Height, NDVI) from the original wide format if possible
    # But wait, the original train.csv is long format.
    # We need to construct metadata features.
    # In this dataset, metadata is effectively "Height" and "NDVI" which were provided as separate files in some versions
    # BUT in this specific User workspace, based on report, 'Height_Ave_cm' and 'Pre_GSHH_NDVI' are features.
    # We need to extract them from the train.csv if they exist, OR match them from a different source 
    # looking at previous scripts (train_tabular_baseline.py) would clarify, but I will assume they are columns 
    # or I need to join with something.
    # Actually, previous report says: "Inputs: Height_Ave_cm, Pre_GSHH_NDVI".
    # Let's check if they are in train.csv?
    # Actually, usually they are.
    
    # Let's inspect the merged df columns in the main block or just assume standard columns
    # Re-reading report: "Missing Values: None found in training metadata".
    # So they are likely in the csv.
    
    # We will assume train.csv has them. If not, we might be missing something fundamental, 
    # but since I haven't seen train.csv fully, I will add a check.
    
    return merged_df

def train_exp6():
    # We need to join with Training Metadata
    # Since I don't have the file view of train.csv, I'll rely on the fact that existing scripts worked.
    # I'll load the full train.csv and pivot properly
    
    df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV))
    
    # Check if metadata is in the rows or columns
    # In this comp, typically metadata is repeated per row or in a separate file.
    # Let's try to pivot metadata too if it exists.
    
    # Attempt to load tabular baseline logic to get metadata
    # Simplified: Just group by image_path and take first of metadata columns
    # Potential metadata cols: 'Height_Ave_cm', 'Pre_GSHH_NDVI', 'Zadoks', 'State', 'Species'
    
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI'] # The key ones
    
    # Pivot Targets
    wide_df = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Extract Metadata (taking first value per image)
    # We need to make sure we don't lose them.
    # If they are columns in df:
    if set(meta_cols).issubset(df.columns):
        meta_df = df.groupby('image_path')[meta_cols].first().reset_index()
        wide_df = pd.merge(wide_df, meta_df, on='image_path')
    
    # Merge with KMeans Features
    feat_df = pd.read_csv(FEATURES_CSV)
    full_df = pd.merge(wide_df, feat_df, on='image_path')
    
    print(f"Data Loaded. Shape: {full_df.shape}")
    
    # Features to use: Metadata + KMeans
    # Exclude IDs and Targets
    drop_cols = ['image_path'] + TARGETS
    feature_cols = [c for c in full_df.columns if c not in drop_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overall_rmse = []
    
    for target in TARGETS:
        print(f"\nTraining for {target}...")
        y = full_df[target].values
        X = full_df[feature_cols].values
        
        target_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            target_rmses.append(rmse)
        
        avg_rmse = np.mean(target_rmses)
        print(f"  -> Avg RMSE: {avg_rmse:.4f}")
        overall_rmse.append(avg_rmse)
        
        # Train Final Model on All Data
        final_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X, y)
        final_model.save_model(os.path.join(OUTPUT_DIR, f'xgb_{target}.json'))
        
    print(f"\nOverall Validation RMSE: {np.mean(overall_rmse):.4f}")

if __name__ == "__main__":
    train_exp6()
