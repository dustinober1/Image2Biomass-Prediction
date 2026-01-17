
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
FEATURES_TRAIN = 'models/features_v1/features_train.csv'
FEATURES_TEST = 'models/features_v1/features_test.csv'
OUTPUT_DIR = 'models/exp2_visual_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_and_predict():
    print("Loading data for Experiment 2 (Visual Features)...")
    
    # 1. Load targets
    df_train_targets = pd.read_csv(TRAIN_CSV)
    # Pivot targets
    targets_pivot = df_train_targets.pivot(index='image_path', columns='target_name', values='target').reset_index()
    
    # 2. Load features
    df_features_train = pd.read_csv(FEATURES_TRAIN)
    df_features_test = pd.read_csv(FEATURES_TEST)
    
    # 3. Merge
    # Ensure image_path formats match. 
    # train.csv has "train/ID..." and features_train also has "train/ID...".
    
    full_train = pd.merge(targets_pivot, df_features_train, on='image_path')
    print(f"Merged Train Shape: {full_train.shape}")
    
    # Feature columns (all except targets and image_path)
    feature_cols = [c for c in df_features_train.columns if c != 'image_path']
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Cross Validation and Training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model_preds_test = np.zeros((len(df_features_test), len(TARGETS)))
    
    val_metrics = []
    
    print("\n--- Training XGBoost on Visual Features ---")
    
    for i, target in enumerate(TARGETS):
        print(f"Training for {target}...")
        
        X = full_train[feature_cols].values
        y = full_train[target].values
        
        cv_rmse = []
        cv_r2 = []
        
        # Test preds for this target
        target_test_preds = np.zeros(len(df_features_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Simple XGBoost
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds_val = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds_val))
            r2 = r2_score(y_val, preds_val)
            
            cv_rmse.append(rmse)
            cv_r2.append(r2)
            
            # Predict on Test (Average over folds)
            target_test_preds += model.predict(df_features_test[feature_cols].values) / kf.get_n_splits()
            
        avg_rmse = np.mean(cv_rmse)
        avg_r2 = np.mean(cv_r2)
        print(f"  Avg RMSE: {avg_rmse:.4f} | Avg R2: {avg_r2:.4f}")
        val_metrics.append({'target': target, 'rmse': avg_rmse, 'r2': avg_r2})
        
        model_preds_test[:, i] = target_test_preds
        
    print("\n--- Overall Performance (Exp 2) ---")
    ov_rmse = np.mean([m['rmse'] for m in val_metrics])
    ov_r2 = np.mean([m['r2'] for m in val_metrics])
    print(f"Overall RMSE: {ov_rmse:.4f}")
    print(f"Overall R2:   {ov_r2:.4f}")
    
    # Save Submission
    submission = []
    
    # Need to match sample_id format: ID_____target
    # features_test has "test/ID1001187975.jpg". We need to extract the ID.
    
    for idx, row in df_features_test.iterrows():
        path = row['image_path'] # e.g. test/ID123.jpg
        # Extract ID
        file_name = os.path.basename(path) # ID123.jpg
        sample_id_base = os.path.splitext(file_name)[0] # ID123
        
        for i, target in enumerate(TARGETS):
            pred_value = model_preds_test[idx, i]
            # ID123__Dry_Green_g
            # Actually, standard submission format needs checking.
            # But based on test.csv content: "ID1001187975__Dry_Clover_g"
            
            sample_id = f"{sample_id_base}__{target}"
            submission.append({'sample_id': sample_id, 'target': max(0, pred_value)}) # Relu
            
    sub_df = pd.DataFrame(submission)
    out_file = os.path.join(OUTPUT_DIR, 'submission_exp2.csv')
    sub_df.to_csv(out_file, index=False)
    print(f"Saved submission to {out_file}")

if __name__ == "__main__":
    train_and_predict()
