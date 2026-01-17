
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Config - Must match train script
DATA_DIR = 'csiro-biomass'
OUTPUT_DIR = 'models/proxy_model_v1'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROXY_TARGETS = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
BATCH_SIZE = 32

class ProxyPredictor(nn.Module):
    def __init__(self):
        super(ProxyPredictor, self).__init__()
        # Use a solid backbone
        self.cnn = models.resnet18(weights=None) # Load weights later
        
        # Replace the final layer
        in_features = self.cnn.fc.in_features
        
        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, len(PROXY_TARGETS)) 
        )
        
    def forward(self, x):
        return self.cnn(x)

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        full_path = os.path.join(DATA_DIR, path)
        try:
            image = Image.open(full_path).convert('RGB')
        except:
             image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
            
        return image, path

def run_inference():
    print("Running Proxy Inference from Checkpoint...")
    
    # Load Test Data
    try:
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['image_path'])
        unique_paths = test_df['image_path'].unique()
    except Exception as e:
        print(f"Error loading test.csv: {e}")
        # Try listing dir as fallback
        test_dir = os.path.join(DATA_DIR, 'test')
        if os.path.exists(test_dir):
            unique_paths = [os.path.join('test', f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
        else:
            print("No test data found.")
            return

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = InferenceDataset(unique_paths, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    model = ProxyPredictor().to(DEVICE)
    weights_path = os.path.join(OUTPUT_DIR, 'proxy_best.pth')
    
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.cpu().numpy()
            
            for p, val in zip(paths, preds):
                results.append({
                    'image_path': p,
                    'Predicted_Height': val[0],
                    'Predicted_NDVI': val[1]
                })
                
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, 'proxy_test_metadata.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved inference results to {out_path}")

if __name__ == "__main__":
    run_inference()
