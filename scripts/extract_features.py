
import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_DIR = 'models/features_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_features(img_path):
    full_path = os.path.join(DATA_DIR, img_path)
    if not os.path.exists(full_path):
        return None
    
    try:
        img = Image.open(full_path).convert('RGB')
        img_arr = np.array(img).astype(np.float32) / 255.0 # Normalize 0-1
        
        # Split channels
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        
        # 1. Color Means and Stds
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        std_r = np.std(r)
        std_g = np.std(g)
        std_b = np.std(b)
        
        # 2. Vegetation Indices
        # ExG = 2g - r - b
        exg = 2*g - r - b
        mean_exg = np.mean(exg)
        
        # ExR = 1.4r - g
        exr = 1.4*r - g
        mean_exr = np.mean(exr)
        
        # CIVE = 0.441r - 0.811g + 0.385b + 18.78 (ignoring constant for correlation)
        cive = 0.441*r - 0.811*g + 0.385*b
        mean_cive = np.mean(cive)
        
        # NDVI-like (using Red and Green since we don't have NIR in standard RGB, though some cameras do)
        # GRVI = (Green - Red) / (Green + Red)
        denom = (g + r + 1e-6)
        grvi = (g - r) / denom
        mean_grvi = np.mean(grvi)
        
        # 3. Texture / Structure proxies
        # Convert to grayscale rough approx
        gray = 0.299*r + 0.587*g + 0.114*b
        
        # Contrast (standard deviation of gray)
        contrast = np.std(gray)
        
        # Fraction of "Green" Pixels (ExG > 0 threshold)
        green_frac = np.mean(exg > 0.05) # threshold 0.05 just to be safe above noise
        
        return {
            'image_path': img_path,
            'feat_Mean_R': mean_r,
            'feat_Mean_G': mean_g,
            'feat_Mean_B': mean_b,
            'feat_Std_R': std_r,
            'feat_Std_G': std_g,
            'feat_Std_B': std_b,
            'feat_Mean_ExG': mean_exg,
            'feat_Mean_ExR': mean_exr,
            'feat_Mean_CIVE': mean_cive,
            'feat_Mean_GRVI': mean_grvi,
            'feat_Contrast': contrast,
            'feat_Green_Frac': green_frac
        }
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def extract_for_csv(csv_path, name):
    print(f"Extracting features for {name} ({csv_path})...")
    df = pd.read_csv(csv_path)
    
    # Get unique images
    image_paths = df['image_path'].unique()
    print(f"Found {len(image_paths)} unique images.")
    
    results = []
    
    # Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_features, path): path for path in image_paths}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0:
                print(f"Processed {i}/{len(image_paths)}")
            res = future.result()
            if res:
                results.append(res)
                
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, f'features_{name}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return results_df

if __name__ == "__main__":
    extract_for_csv(TRAIN_CSV, 'train')
    if os.path.exists(TEST_CSV):
        extract_for_csv(TEST_CSV, 'test')
