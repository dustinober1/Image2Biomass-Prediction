
import pandas as pd
import numpy as np
import os
from PIL import Image

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')

def analyze_tabular():
    print("--- Tabular Data Analysis ---")
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return

    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {TRAIN_CSV}, Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution Summary:")
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    # train.csv is in long format: each row has 'target_name' and 'target' value.
    # We need to pivot or just look at 'target' statistics grouped by 'target_name'.
    
    # Check if format is long or wide. Based on Description:
    # "target_name — Biomass component name for this row... target — Ground-truth biomass value"
    # It is LONG format. One row per target per image? Or one row per image (sample_id) per target?
    # Let's inspect sample_id duplicates.
    
    unique_samples = df['sample_id'].nunique()
    print(f"\nUnique sample_ids: {unique_samples}")
    print(f"Total rows: {len(df)}")
    
    # Pivot to wide format for correlation analysis
    try:
        df_wide = df.pivot(index='sample_id', columns='target_name', values='target')
        # Join with metadata (take first entry per sample_id for metadata cols)
        metadata_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State', 'Species']
        meta = df.groupby('sample_id')[metadata_cols].first()
        df_full = df_wide.join(meta)
        
        print("\nCorrelation Matrix (Targets vs Features):")
        corr_cols = target_cols + ['Height_Ave_cm', 'Pre_GSHH_NDVI']
        # Filter only existing columns
        existing_cols = [c for c in corr_cols if c in df_full.columns]
        corr = df_full[existing_cols].corr()
        print(corr)
        
        print("\nStatistics for targets (Wide Format):")
        print(df_full[target_cols].describe())

    except Exception as e:
        print(f"Error pivoting or calculating stats: {e}")
        # Fallback to simple groupby
        print("\nStats by target_name:")
        print(df.groupby('target_name')['target'].describe())

    return df

def analyze_images(df):
    print("\n--- Image Analysis ---")
    if df is None: return

    # Check valid images
    # sample_id to image_path mapping
    # Check if 'image_path' exists
    if 'image_path' not in df.columns:
        print("image_path column not found.")
        return

    # specific check
    # image paths in csv might be relative to DATA_DIR or just 'train/'?
    # Description says: "image_path — Relative path to the image (e.g., images/ID1098771283.jpg)."
    # But files section says "train/ Directory containing training images".
    
    # We will try to find where the images are.
    # checking first distinct image path
    first_path = df['image_path'].iloc[0]
    print(f"Sample image path in CSV: {first_path}")
    
    # Full path construction
    # Try joining DATA_DIR + first_path
    test_path_1 = os.path.join(DATA_DIR, first_path)
    # Try joining DATA_DIR + 'train' + filename if 'train' is not in path
    
    print(f"Checking access to: {test_path_1}")
    if os.path.exists(test_path_1):
        print("Path structure confirmed: csiro-biomass/ + image_path")
        valid_path = getattr(df, 'image_path').apply(lambda x: os.path.join(DATA_DIR, x))
    else:
        # Check if we need to adjust
        print("Direct join failed. Checking dir contents...")
        # (This is just debugging info for us)
        pass

    # Sample 5 images
    print("\nSampling 5 images for dimensions check:")
    try:
        # Get unique image paths
        unique_paths = df['image_path'].unique()[:5]
        for p in unique_paths:
            full_p = os.path.join(DATA_DIR, p)
            if os.path.exists(full_p):
                with Image.open(full_p) as img:
                    print(f"{p}: Size={img.size}, Mode={img.mode}, Format={img.format}")
            else:
                print(f"{p}: Not found at {full_p}")
    except Exception as e:
        print(f"Image check error: {e}")

if __name__ == "__main__":
    df = analyze_tabular()
    analyze_images(df)
