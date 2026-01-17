
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import joblib

# Configuration
DATA_DIR = 'csiro-biomass'
TEST_CSV = os.path.join(DATA_DIR, 'test.csv') # This might just be metadata or list of IDs
MODEL_DIR = 'models/tabular_final'
OUTPUT_FILE = 'submission.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def load_test_data():
    print("Loading test data...")
    # NOTE: user has not shown the structure of test.csv. Assuming it has the same metadata columns.
    # If not, we might need to look at images.
    # But for tabular model we NEED tabular metadata in test.csv.
    
    df = pd.read_csv(TEST_CSV)
    print(f"Test data shape: {df.shape}")
    
    # Check for missing columns and impute if necessary
    required_metadata = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State', 'Species']
    missing_cols = [c for c in required_metadata if c not in df.columns]
    
    if missing_cols:
        print(f"WARNING: Test data missing columns: {missing_cols}. Imputing with training stats.")
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        
        # Impute Continuous with Mean
        if 'Height_Ave_cm' in missing_cols:
            mean_height = train_df['Height_Ave_cm'].mean()
            df['Height_Ave_cm'] = mean_height
            print(f"  Imputed Height_Ave_cm: {mean_height:.2f}")
            
        if 'Pre_GSHH_NDVI' in missing_cols:
            mean_ndvi = train_df['Pre_GSHH_NDVI'].mean()
            df['Pre_GSHH_NDVI'] = mean_ndvi
            print(f"  Imputed Pre_GSHH_NDVI: {mean_ndvi:.2f}")
            
        # Impute Categorical with Mode
        if 'State' in missing_cols:
            mode_state = train_df['State'].mode()[0]
            df['State'] = mode_state
            print(f"  Imputed State: {mode_state}")
            
        if 'Species' in missing_cols:
            mode_species = train_df['Species'].mode()[0]
            df['Species'] = mode_species
            print(f"  Imputed Species: {mode_species}")
            
    # Load Encoders
    le_state = joblib.load(os.path.join(MODEL_DIR, 'le_state.pkl'))
    le_species = joblib.load(os.path.join(MODEL_DIR, 'le_species.pkl'))
    
    # Transform
    df['State_Encoded'] = le_state.transform(df['State'])
    df['Species_Encoded'] = le_species.transform(df['Species'])
    
    features = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']

    return df, features

def generate_submission():
    df, features = load_test_data()
    X_test = df[features]
    
    # Prepare Output DataFrame
    # Needs sample_id, and 5 target columns? 
    # Usually submission format is: sample_id, target_name, target (long) OR sample_id, t1, t2...
    # The sample_submission.csv format was not explicitly shown but standard Kaggle is often long or wide.
    # Based on train.csv being long, submission might be long too.
    # Let's check sample_submission.csv first if we could.
    # For now, generate WIDE and then melt if needed.
    
    results = pd.DataFrame()
    results['sample_id'] = df['sample_id'] # Assuming sample_id exists in test.csv
    
    for target in TARGETS:
        print(f"Predicting {target}...")
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(MODEL_DIR, f"xgb_{target}.json"))
        
        preds = model.predict(X_test)
        results[target] = preds
        
    # Check if we need to reshape.
    # "The mapping is shown as follows in the format [URI] -> [CorpusName]" - incorrect context
    # Looking at train.csv: sample_id, target_name, target
    # If submission needs to be this format:
    
    # For safety, let's create a LONG version too or check sample_submission.
    # I will stick to WIDE for now as it's easier to verify.
    # If the user provides sample_submission structure, I will adapt.
    
    print(f"Saving submission to {OUTPUT_FILE}...")
    results.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_submission()
