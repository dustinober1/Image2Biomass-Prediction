
import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
import time

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_DIR = 'models/features_kmeans'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 3 # Soil/Shadow, Dead/Dry, Green
RESIZE_DIM = (256, 128) # Faster K-Means

def get_kmeans_features(img_path):
    full_path = os.path.join(DATA_DIR, img_path)
    if not os.path.exists(full_path):
        return None
    
    try:
        # Load and resize for speed
        img = Image.open(full_path).convert('RGB')
        img_small = img.resize(RESIZE_DIM)
        img_arr = np.array(img_small).astype(np.float32) / 255.0
        
        # Flatten: (N_pixels, 3)
        h, w, c = img_arr.shape
        pixels = img_arr.reshape(-1, 3)
        
        # K-Means
        # Use MiniBatch for speed on many images
        kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=2048, n_init=3)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_ # (3, 3) -> RGB
        
        # Calculate cluster properties
        cluster_props = []
        for i in range(N_CLUSTERS):
            # Mask for this cluster
            mask = (labels == i)
            count = np.sum(mask)
            fraction = count / len(pixels)
            
            # Center RGB
            r, g, b = centers[i]
            
            # Texture: Std Dev of pixels in this cluster
            cluster_pixels = pixels[mask]
            if len(cluster_pixels) > 1:
                r_std, g_std, b_std = np.std(cluster_pixels, axis=0)
            else:
                r_std, g_std, b_std = 0.0, 0.0, 0.0
            
            # Calculate "Greenness" for sorting
            # ExG = 2G - R - B
            exg = 2*g - r - b
            
            cluster_props.append({
                'id': i,
                'fraction': fraction,
                'R': r, 'G': g, 'B': b,
                'R_std': r_std, 'G_std': g_std, 'B_std': b_std,
                'ExG': exg
            })
            
        # Sort clusters by ExG (Ascending: Soil -> Dead -> Green)
        # Assuming: 
        #   Lowest ExG -> Soil/Shadow/Background 
        #   Middle ExG -> Dead/Dry matter (brownish)
        #   Highest ExG -> Green matter
        cluster_props.sort(key=lambda x: x['ExG'])
        
        features = {'image_path': img_path}
        
        # Flatten carefully
        prefix_map = {0: 'Soil', 1: 'Dead', 2: 'Green'}
        
        for i, props in enumerate(cluster_props):
            p = prefix_map.get(i, f'C{i}')
            features[f'KM_{p}_Frac'] = props['fraction']
            features[f'KM_{p}_R'] = props['R']
            features[f'KM_{p}_G'] = props['G']
            features[f'KM_{p}_B'] = props['B']
            features[f'KM_{p}_R_std'] = props['R_std']
            features[f'KM_{p}_G_std'] = props['G_std']
            features[f'KM_{p}_B_std'] = props['B_std']
            features[f'KM_{p}_ExG'] = props['ExG']
            
        return features
        
    except Exception as e:
        print(f"Error {img_path}: {e}")
        return None

def extract_for_csv(csv_path, name):
    print(f"Extracting K-Means features for {name}...")
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].unique()
    
    results = []
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: # CPU bound-ish
        futures = {executor.submit(get_kmeans_features, path): path for path in image_paths}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 20 == 0:
                print(f"Processed {i}/{len(image_paths)}")
            res = future.result()
            if res:
                results.append(res)
                
    print(f"Finished in {time.time() - start_time:.1f}s")
    
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, f'features_kmeans_{name}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_df

if __name__ == "__main__":
    extract_for_csv(TRAIN_CSV, 'train')
    if os.path.exists(TEST_CSV):
        extract_for_csv(TEST_CSV, 'test')
