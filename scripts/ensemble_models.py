
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import joblib

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# Paths
MODEL_TABULAR = 'models/tabular_final'
MODEL_EFFNET = 'models/exp4_efficientnet_tta/effnet_best.pth'
MODEL_KMEANS = 'models/exp6_kmeans'
MODEL_PROXY = 'models/proxy_model_v1/proxy_best.pth'
FEATS_KMEANS_TRAIN = 'models/features_kmeans/features_kmeans_train.csv'
FEATS_KMEANS_TEST = 'models/features_kmeans/features_kmeans_test.csv'

def load_data():
    df = pd.read_csv(TRAIN_CSV)
    
    # Pivot Targets
    wide_df = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Metadata
    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State', 'Species']
    meta_df = df.groupby('image_path')[meta_cols].first().reset_index()
    
    wide_df = pd.merge(wide_df, meta_df, on='image_path')
    
    # KMeans Features
    km_df = pd.read_csv(FEATS_KMEANS_TRAIN)
    wide_df = pd.merge(wide_df, km_df, on='image_path')
    
    return wide_df

def get_proxy_height(images, model):
    # Simplified proxy inference for validation set
    # In practice, we'd use the proxy script, but for speed in this ensemble script:
    model.eval()
    with torch.no_grad():
        preds = model(images.to(DEVICE))
        return preds[:, 0].cpu().numpy() # 0 is Height_Ave_cm usually

class ProxyPredictor(nn.Module):
    def __init__(self):
        super(ProxyPredictor, self).__init__()
        from torchvision import models
        self.cnn = models.resnet18(weights=None)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Height, NDVI
        )
    def forward(self, x):
        return self.cnn(x)

def run_ensemble():
    full_df = load_data()
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    print(f"Validation Set Size: {len(val_df)}")
    
    # --- 1. Tabular Model Predictions ---
    # We use Predicted Height for the ensemble to match test conditions
    # Load Proxy
    proxy = ProxyPredictor().to(DEVICE)
    proxy.load_state_dict(torch.load(MODEL_PROXY, map_location=DEVICE))
    proxy.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Running Proxy Inference on Val Set...")
    pred_heights = []
    for path in val_df['image_path']:
        img = Image.open(os.path.join(DATA_DIR, path)).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = proxy(img_t.to(DEVICE))
            pred_heights.append(out.cpu().numpy()[0, 0])
    
    val_df_proxy = val_df.copy()
    val_df_proxy['Height_Ave_cm'] = pred_heights # Override with predicted
    
    # Encodings
    le_state = joblib.load(os.path.join(MODEL_TABULAR, 'le_state.pkl'))
    le_species = joblib.load(os.path.join(MODEL_TABULAR, 'le_species.pkl'))
    val_df_proxy['State_Encoded'] = le_state.transform(val_df_proxy['State'])
    val_df_proxy['Species_Encoded'] = le_species.transform(val_df_proxy['Species'])
    
    tab_features = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']
    
    tab_preds = []
    for target in TARGETS:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(MODEL_TABULAR, f'xgb_{target}.json'))
        tab_preds.append(model.predict(val_df_proxy[tab_features]))
    tab_preds = np.array(tab_preds).T # (N, 5)
    
    # --- 2. EfficientNet Predictions ---
    print("Running EffNet Inference on Val Set...")
    effnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5).to(DEVICE)
    effnet.load_state_dict(torch.load(MODEL_EFFNET, map_location=DEVICE))
    effnet.eval()
    
    eff_preds = []
    for path in val_df['image_path']:
        img = Image.open(os.path.join(DATA_DIR, path)).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = effnet(img_t.to(DEVICE))
            eff_preds.append(out.cpu().numpy()[0])
    eff_preds = np.array(eff_preds) # (N, 5)
    
    # --- 3. K-Means Predictions ---
    print("Running K-Means Tabular Inference on Val Set...")
    km_feats = [c for c in val_df.columns if 'KM_' in c]
    km_all_feats = ['Height_Ave_cm', 'Pre_GSHH_NDVI'] + km_feats
    
    km_preds = []
    for target in TARGETS:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(MODEL_KMEANS, f'xgb_{target}.json'))
        km_preds.append(model.predict(val_df_proxy[km_all_feats]))
    km_preds = np.array(km_preds).T # (N, 5)
    
    # --- Optimization ---
    y_true = val_df[TARGETS].values
    
    def objective(w):
        # w is [w1, w2, w3] per target or global
        # Making it per target for maximum flexibility
        # w shape (15,) -> 3 weights * 5 targets
        weights = w.reshape(5, 3)
        total_mse = 0
        for i in range(5):
            p = weights[i, 0]*tab_preds[:, i] + weights[i, 1]*eff_preds[:, i] + weights[i, 2]*km_preds[:, i]
            total_mse += mean_squared_error(y_true[:, i], p)
        return total_mse / 5
    
    initial_w = np.ones(15) / 3
    bounds = [(0, 1)] * 15
    res = minimize(objective, initial_w, bounds=bounds, method='L-BFGS-B')
    opt_weights = res.x.reshape(5, 3)
    
    print("\nOptimal Weights (Tabular, EffNet, KMeans):")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: {opt_weights[i]}")
        
    final_preds = np.zeros_like(y_true)
    for i in range(5):
        final_preds[:, i] = opt_weights[i, 0]*tab_preds[:, i] + opt_weights[i, 1]*eff_preds[:, i] + opt_weights[i, 2]*km_preds[:, i]
        
    ensemble_rmse = np.sqrt(mean_squared_error(y_true, final_preds))
    print(f"\nEnsemble Validation RMSE: {ensemble_rmse:.4f}")
    
    # Save Weights
    np.save('models/ensemble_weights.npy', opt_weights)
    
    # --- Generate Test Submission ---
    print("\nGenerating Test Submission...")
    test_csv_df = pd.read_csv(TEST_CSV)
    test_paths = test_csv_df['image_path'].unique()
    
    # Test Metadata (using Training Mean for Height and NDVI since test has none)
    # AND categorical mode
    train_meta = full_df[['Height_Ave_cm', 'Pre_GSHH_NDVI']].mean()
    # For Height, we should use Proxy. For NDVI, we use Mean.
    
    # Test KMeans
    test_km = pd.read_csv(FEATS_KMEANS_TEST)
    
    submission = []
    
    # Prepare individual pred dicts for mapping
    # (Same inference logic as above but for test)
    for path in test_paths:
        img = Image.open(os.path.join(DATA_DIR, path)).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            proxy_out = proxy(img_t.to(DEVICE)).cpu().numpy()[0]
            val_eff = effnet(img_t.to(DEVICE)).cpu().numpy()[0]
            
        p_height = proxy_out[0]
        # p_ndvi = proxy_out[1] # Proxy NDVI is weak, maybe use train mean? 
        # Report said NDVI proxy was weak.
        
        # Metadata row for test
        test_meta_row = pd.DataFrame([{
            'Height_Ave_cm': p_height,
            'Pre_GSHH_NDVI': train_meta['Pre_GSHH_NDVI'], # Use mean for NDVI
            'State_Encoded': le_state.transform(['Vic'])[0], # Mode approx
            'Species_Encoded': le_species.transform(['Lucerne'])[0] # Mode approx
        }])
        
        # KMeans row
        km_row = test_km[test_km['image_path'] == path].drop(columns=['image_path'])
        km_meta_row = pd.concat([test_meta_row, km_row.reset_index(drop=True)], axis=1)
        
        p_tab = []
        p_km = []
        for i, target in enumerate(TARGETS):
            m_tab = xgb.XGBRegressor()
            m_tab.load_model(os.path.join(MODEL_TABULAR, f'xgb_{target}.json'))
            v_tab = m_tab.predict(test_meta_row[tab_features])[0]
            p_tab.append(v_tab)
            
            m_km = xgb.XGBRegressor()
            m_km.load_model(os.path.join(MODEL_KMEANS, f'xgb_{target}.json'))
            v_km = m_km.predict(km_meta_row[km_all_feats])[0]
            p_km.append(v_km)
            
        # Ensemble
        ensemble_p = []
        for i in range(5):
            val = opt_weights[i,0]*p_tab[i] + opt_weights[i,1]*val_eff[i] + opt_weights[i,2]*p_km[i]
            ensemble_p.append(max(0, val))
            
        # Map back to sample_ids
        for i, target in enumerate(TARGETS):
            s_id = f"{path.split('/')[-1].replace('.jpg','')}__{target}"
            submission.append({'sample_id': s_id, 'target': ensemble_p[i]})
            
    sub_df = pd.DataFrame(submission)
    # Ensure it matches test_csv order
    sub_df = pd.merge(test_csv_df[['sample_id']], sub_df, on='sample_id')
    sub_df.to_csv('submission_ensemble.csv', index=False)
    print("Saved submission_ensemble.csv")

if __name__ == "__main__":
    run_ensemble()
