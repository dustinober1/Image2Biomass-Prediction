
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
PROXY_METADATA_CSV = 'models/proxy_model_v1/proxy_test_metadata.csv'
TABULAR_MODEL_DIR = 'models/tabular_final'
OUTPUT_DIR = 'models/exp1_proxy_metadata'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
FEATURE_COLS = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']

def run_inference():
    print("Running Inference using Proxy Metadata...")
    
    # 1. Load Proxy Metadata
    if not os.path.exists(PROXY_METADATA_CSV):
        print("Proxy metadata not found.")
        return
    
    proxy_df = pd.read_csv(PROXY_METADATA_CSV)
    print(f"Loaded {len(proxy_df)} test images with proxy metadata.")
    
    # 2. Load Training Data to compute modes for State/Species
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Compute modes
    mode_state = train_df['State'].mode()[0]
    mode_species = train_df['Species'].mode()[0]
    print(f"Imputing State='{mode_state}', Species='{mode_species}'")
    
    # 3. Prepare Test Features
    # Fill State/Species
    proxy_df['State'] = mode_state
    proxy_df['Species'] = mode_species
    
    # Load Encoders
    le_state = joblib.load(os.path.join(TABULAR_MODEL_DIR, 'le_state.pkl'))
    le_species = joblib.load(os.path.join(TABULAR_MODEL_DIR, 'le_species.pkl'))
    
    # Encode
    # Note: If mode is not in encoder (unlikely since it comes from train), handle error
    try:
        proxy_df['State_Encoded'] = le_state.transform(proxy_df['State'])
        proxy_df['Species_Encoded'] = le_species.transform(proxy_df['Species'])
    except Exception as e:
        print(f"Encoding error: {e}")
        return

    # Map columns to match XGBoost expectations
    # Proxy gives 'Predicted_Height', 'Predicted_NDVI'
    proxy_df['Height_Ave_cm'] = proxy_df['Predicted_Height']
    proxy_df['Pre_GSHH_NDVI'] = proxy_df['Predicted_NDVI']
    
    X_test = proxy_df[FEATURE_COLS]
    
    # 4. Predict
    submission = []
    
    for target in TARGETS:
        print(f"Predicting {target}...")
        model_path = os.path.join(TABULAR_MODEL_DIR, f"xgb_{target}.json")
        
        # Load XGBoost Model
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        preds = model.predict(X_test)
        
        # Add to submission list
        # We need "sample_id" which is ImageID__TargetName
        # proxy_df has 'image_path'. Extract ID.
        
        for idx, row in proxy_df.iterrows():
            img_path = row['image_path'] # e.g. test/ID123.jpg
            fname = os.path.basename(img_path)
            sample_id_base = os.path.splitext(fname)[0]
            
            s_id = f"{sample_id_base}__{target}"
            val = max(0, preds[idx])
            submission.append({'sample_id': s_id, 'target': val})
            
    sub_df = pd.DataFrame(submission)
    out_file = os.path.join(OUTPUT_DIR, 'submission_exp1.csv')
    sub_df.to_csv(out_file, index=False)
    print(f"Saved submission to {out_file}")

if __name__ == "__main__":
    run_inference()
