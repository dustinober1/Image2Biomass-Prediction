
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/tabular_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
N_ITER = 20 # Number of parameter settings that are sampled
CV = 3 # Cross-validation folds

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

def train_feature_model(df, features, target_name):
    print(f"\nTraining for {target_name}...")
    
    # Filter valid data
    valid_mask = df[target_name].notna()
    X = df.loc[valid_mask, features]
    y = df.loc[valid_mask, target_name]
    
    print(f"  Samples: {len(X)}")
    
    # Parameter Grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 300, 500, 1000],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', nthread=1, random_state=42)
    
    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=CV,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X, y)
    
    print(f"  Best Params: {search.best_params_}")
    print(f"  Best CV RMSE: {-search.best_score_:.4f}")
    
    # Train final model on all data with best params
    best_model = search.best_estimator_
    
    # Save
    best_model.save_model(os.path.join(OUTPUT_DIR, f"xgb_{target_name}.json"))
    
    return best_model, -search.best_score_

def train_all():
    df, features = load_and_preprocess()
    
    results = {}
    
    for target in TARGETS:
        model, cv_rmse = train_feature_model(df, features, target)
        results[target] = cv_rmse
        
    print("\n--- Final Cross-Validation Results ---")
    avg_rmse = 0
    for t, rmse in results.items():
        print(f"{t}: RMSE={rmse:.4f}")
        avg_rmse += rmse
    print(f"Average RMSE: {avg_rmse/len(TARGETS):.4f}")

if __name__ == "__main__":
    train_all()
