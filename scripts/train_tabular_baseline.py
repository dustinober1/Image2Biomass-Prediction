
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/tabular_baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV)
    
    metadata_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State', 'Species']
    targets_long = df[['sample_id', 'target_name', 'target']]
    
    # Pivot targets
    targets_wide = targets_long.pivot(index='sample_id', columns='target_name', values='target').reset_index()
    
    # Get metadata
    meta_df = df.groupby('sample_id')[metadata_cols].first().reset_index()
    
    # Merge
    full_df = pd.merge(targets_wide, meta_df, on='sample_id')
    print(f"Prepared wide dataset: {full_df.shape}")
    
    # Encoding
    le_state = LabelEncoder()
    full_df['State_Encoded'] = le_state.fit_transform(full_df['State'])
    
    le_species = LabelEncoder()
    full_df['Species_Encoded'] = le_species.fit_transform(full_df['Species'])
    
    feature_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']
    
    joblib.dump(le_state, os.path.join(OUTPUT_DIR, 'le_state.pkl'))
    joblib.dump(le_species, os.path.join(OUTPUT_DIR, 'le_species.pkl'))
    
    return full_df, feature_cols

def train_model(df, features):
    print("\nStarting Training...")
    results = {}
    
    overall_metrics = {'rmse': [], 'mae': [], 'r2': []}
    
    for target in TARGETS:
        print(f"Training for {target}...")
        
        # Filter out missing targets
        valid_mask = df[target].notna()
        X = df.loc[valid_mask, features]
        y = df.loc[valid_mask, target]
        
        print(f"  Samples with valid {target}: {len(X)}")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Debugging: Inspect raw data before conversion
        print(f"DEBUG: Feature columns: {features}")
        # print("DEBUG: X_train head:\n", X_train.head())
        # print("DEBUG: y_train head:\n", y_train.head())
        
        try:
            # Force conversion
            X_train_np = np.array(X_train, dtype=np.float32)
            X_val_np = np.array(X_val, dtype=np.float32)
            y_train_np = np.array(y_train, dtype=np.float32)
            y_val_np = np.array(y_val, dtype=np.float32)
        except Exception as e:
            print(f"CRITICAL ERROR in data conversion: {e}")
            raise e

        # Use native XGBoost API
        dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
        dval = xgb.DMatrix(X_val_np, label=y_val_np)
        
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 5,
            'eval_metric': 'rmse',
            'nthread': 1 
        }
        
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        print("Training finished.")
        
        preds = bst.predict(dval)
        
        rmse = np.sqrt(mean_squared_error(y_val_np, preds))
        mae = mean_absolute_error(y_val_np, preds)
        
        # Calculate R2 manually to be safe
        ss_res = np.sum((y_val_np - preds) ** 2)
        ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        results[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        overall_metrics['rmse'].append(rmse)
        overall_metrics['mae'].append(mae)
        overall_metrics['r2'].append(r2)
        
        bst.save_model(os.path.join(OUTPUT_DIR, f"xgb_{target}.json"))

    print("\n--- Overall Baseline Performance (Average) ---")
    print(f"Avg RMSE: {np.mean(overall_metrics['rmse']):.4f}")
    print(f"Avg MAE:  {np.mean(overall_metrics['mae']):.4f}")
    print(f"Avg R2:   {np.mean(overall_metrics['r2']):.4f}")

    return results

if __name__ == "__main__":
    df, features = load_and_preprocess()
    train_model(df, features)
