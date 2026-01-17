
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/stacking/meta'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def train_meta():
    # Load OOFs
    try:
        oof_tab = pd.read_csv('models/stacking/tabular/oof_tabular.csv')
        oof_km = pd.read_csv('models/stacking/kmeans/oof_kmeans.csv')
        oof_eff = pd.read_csv('models/stacking/effnet/oof_effnet.csv')
    except Exception as e:
        print(f"Error loading OOF files: {e}")
        return

    # Merge
    master_oof = pd.merge(oof_tab, oof_km, on='image_path')
    master_oof = pd.merge(master_oof, oof_eff, on='image_path')
    
    # Load Real Targets
    train_df = pd.read_csv(TRAIN_CSV)
    wide_true = train_df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    master_oof = pd.merge(master_oof, wide_true, on='image_path')
    
    oof_final = np.zeros((len(master_oof), len(TARGETS)))
    meta_models = {}

    print("\nTraining Meta-Learner (Ridge) per target...")
    for i, target in enumerate(TARGETS):
        # Features for this meta target: all models' predictions for THIS target
        features = [f'OOF_Tabular_{target}', f'OOF_KMeans_{target}', f'OOF_EffNet_{target}']
        X = master_oof[features].values
        y = master_oof[target].values
        
        # 5-Fold Meta-CV to score the meta-learner itself
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        target_oof_preds = np.zeros(len(master_oof))
        
        for fold, (t_idx, v_idx) in enumerate(kf.split(X)):
            meta = Ridge(alpha=1.0)
            meta.fit(X[t_idx], y[t_idx])
            target_oof_preds[v_idx] = meta.predict(X[v_idx])
            
        oof_final[:, i] = target_oof_preds
        
        # Train final meta-learner on ALL data for inference
        final_meta = Ridge(alpha=1.0)
        final_meta.fit(X, y)
        meta_models[target] = final_meta
        joblib.dump(final_meta, os.path.join(OUTPUT_DIR, f'meta_ridge_{target}.pkl'))
        
        rmse = np.sqrt(mean_squared_error(y, target_oof_preds))
        print(f"  {target}: Meta OOF RMSE = {rmse:.4f} (Weights: {final_meta.coef_})")

    # Overall Score
    overall_rmse = np.sqrt(mean_squared_error(master_oof[TARGETS], oof_final))
    print(f"\nStacked Ensemble Overall OOF RMSE: {overall_rmse:.4f}")

if __name__ == "__main__":
    train_meta()
