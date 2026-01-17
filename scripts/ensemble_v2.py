
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import xgboost as xgb
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# Paths
PATH_TABULAR = 'models/stacking/tabular'
PATH_KMEANS = 'models/stacking/kmeans'
PATH_EFFNET = 'models/stacking/effnet'
PATH_META = 'models/stacking/meta'
FEATS_KMEANS_TEST = 'models/features_kmeans/features_kmeans_test.csv'

def run_inference():
    test_df = pd.read_csv(TEST_CSV)
    test_paths = test_df['image_path'].unique()
    
    # Load Meta-Predictors (Proxy for Height, etc.)
    # We will use the same proxy as Experiment 7 for Height
    # (Assuming proxy_best.pth exists in its previous location)
    class ProxyPredictor(nn.Module):
        def __init__(self):
            super(ProxyPredictor, self).__init__()
            from torchvision import models
            self.cnn = models.resnet18(weights=None)
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(self, x): return self.cnn(x)
    
    proxy = ProxyPredictor().to(DEVICE)
    proxy.load_state_dict(torch.load('models/proxy_model_v1/proxy_best.pth', map_location=DEVICE))
    proxy.eval()
    
    # Load KM Features
    test_km = pd.read_csv(FEATS_KMEANS_TEST)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Meta and Logic Encoders
    le_state = joblib.load(os.path.join(PATH_TABULAR, 'le_state.pkl'))
    le_species = joblib.load(os.path.join(PATH_TABULAR, 'le_species.pkl'))
    
    # Inference loop
    predictions_map = {}
    
    # Pre-load all 5-fold models (optional for speed, or just loop)
    # For now, we loop per path for clarity
    
    for path in test_paths:
        img = Image.open(os.path.join(DATA_DIR, path)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        # 1. Proxy
        with torch.no_grad():
            proxy_out = proxy(img_t).cpu().numpy()[0]
        p_height = proxy_out[0]
        
        # 2. Tabular Base Features
        # Using Vic/Lucerne as mode approx as done previously
        meta_row = pd.DataFrame([{
            'Height_Ave_cm': p_height,
            'Pre_GSHH_NDVI': 0.5, # Train mean approx
            'State_Encoded': le_state.transform(['Vic'])[0],
            'Species_Encoded': le_species.transform(['Lucerne'])[0]
        }])
        
        # 3. KMeans Base Features
        km_row = test_km[test_km['image_path'] == path].drop(columns=['image_path']).reset_index(drop=True)
        # Combine meta and km
        km_meta_row = pd.concat([meta_row[['Height_Ave_cm', 'Pre_GSHH_NDVI']], km_row], axis=1)
        
        # --- Multi-Fold Predictions ---
        # Initialize (N_folds, N_targets) for each type
        all_tab = np.zeros((5, 5))
        all_km = np.zeros((5, 5))
        all_eff = np.zeros((5, 5))
        
        # EffNet Folds
        for f in range(5):
            effnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5).to(DEVICE)
            effnet.load_state_dict(torch.load(os.path.join(PATH_EFFNET, f'effnet_fold{f}.pth'), map_location=DEVICE))
            effnet.eval()
            with torch.no_grad():
                out = effnet(img_t).cpu().numpy()[0]
                all_eff[f] = out
        
        # Tabular/KMeans Folds
        for f in range(5):
            for i, target in enumerate(TARGETS):
                # Tab
                m_tab = xgb.XGBRegressor()
                m_tab.load_model(os.path.join(PATH_TABULAR, f'xgb_{target}_fold{f}.json'))
                all_tab[f, i] = m_tab.predict(meta_row[['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']])[0]
                # KM
                km_feats = [c for c in km_meta_row.columns if 'KM_' in c]
                km_feats_all = ['Height_Ave_cm', 'Pre_GSHH_NDVI'] + km_feats
                m_km = xgb.XGBRegressor()
                m_km.load_model(os.path.join(PATH_KMEANS, f'xgb_{target}_fold{f}.json'))
                all_km[f, i] = m_km.predict(km_meta_row[km_feats_all])[0]
        
        # Average the 5 folds to get the final "base features" for the meta-learner
        final_tab_base = np.mean(all_tab, axis=0) # (5,)
        final_km_base = np.mean(all_km, axis=0)   # (5,)
        final_eff_base = np.mean(all_eff, axis=0) # (5,)
        
        # 4. Meta-Learner Layer
        final_stacked_pred = []
        for i, target in enumerate(TARGETS):
            meta_model = joblib.load(os.path.join(PATH_META, f'meta_ridge_{target}.pkl'))
            # [ Tab_pred, KM_pred, Eff_pred ]
            meta_input = np.array([[final_tab_base[i], final_km_base[i], final_eff_base[i]]])
            stacked_val = meta_model.predict(meta_input)[0]
            final_stacked_pred.append(max(0, stacked_val))
            
        predictions_map[path] = final_stacked_pred
        
    # Format for Submission
    submission = []
    for idx, row in test_df.iterrows():
        path = row['image_path']
        s_id = row['sample_id']
        t_name = row['target_name']
        
        if path in predictions_map:
            t_idx = TARGETS.index(t_name)
            val = predictions_map[path][t_idx]
            submission.append({'sample_id': s_id, 'target': val})
        else:
            submission.append({'sample_id': s_id, 'target': 0})
            
    pd.DataFrame(submission).to_csv('submission_stacking.csv', index=False)
    print("Saved submission_stacking.csv")

if __name__ == "__main__":
    run_inference()
