
import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures
import time

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_DIR = 'models/features_grid'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid Settings
GRID_W = 4
GRID_H = 2

def get_grid_features(img_path):
    full_path = os.path.join(DATA_DIR, img_path)
    if not os.path.exists(full_path):
        return None
    
    try:
        img = Image.open(full_path).convert('RGB')
        img_arr = np.array(img).astype(np.float32) / 255.0
        
        h, w, _ = img_arr.shape
        cell_h = h // GRID_H
        cell_w = w // GRID_W
        
        features = {'image_path': img_path}
        
        for i in range(GRID_H):
            for j in range(GRID_W):
                cell_idx = i * GRID_W + j
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < GRID_H - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < GRID_W - 1 else w
                
                cell = img_arr[y_start:y_end, x_start:x_end]
                
                # Spectral components
                r = cell[:, :, 0]
                g = cell[:, :, 1]
                b = cell[:, :, 2]
                
                # Basic Stats
                features[f'G{cell_idx}_Mean_R'] = np.mean(r)
                features[f'G{cell_idx}_Mean_G'] = np.mean(g)
                features[f'G{cell_idx}_Mean_B'] = np.mean(b)
                
                # Targeted Vegetation Indices (Exp 12)
                # ExG
                exg = 2*g - r - b
                features[f'G{cell_idx}_ExG'] = np.mean(exg)
                
                # VARI = (G - R) / (G + R - B)
                vari = (g - r) / (g + r - b + 1e-6)
                features[f'G{cell_idx}_VARI'] = np.mean(vari)
                
                # GLI = (2G - R - B) / (2G + R + B)
                gli = (2*g - r - b) / (2*g + r + b + 1e-6)
                features[f'G{cell_idx}_GLI'] = np.mean(gli)
                
                # NGRDI = (G - R) / (G + R)
                ngrdi = (g - r) / (g + r + 1e-6)
                features[f'G{cell_idx}_NGRDI'] = np.mean(ngrdi)
                
                # Texture / Structure
                gray = 0.299*r + 0.587*g + 0.114*b
                features[f'G{cell_idx}_Contrast'] = np.std(gray)
                
        return features
        
    except Exception as e:
        print(f"Error {img_path}: {e}")
        return None

def extract_for_csv(csv_path, name):
    print(f"Extracting grid features for {name}...")
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].unique()
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_grid_features, path): path for path in image_paths}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 50 == 0:
                print(f"Processed {i}/{len(image_paths)}")
            res = future.result()
            if res:
                results.append(res)
                
    print(f"Finished in {time.time() - start_time:.1f}s")
    
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, f'features_grid_{name}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_df

if __name__ == "__main__":
    extract_for_csv(TRAIN_CSV, 'train')
    if os.path.exists(TEST_CSV):
        extract_for_csv(TEST_CSV, 'test')
