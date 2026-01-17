
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/stacking/effnet'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30 # Reduced for speed in OOF demonstration
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = df['image_path'].values
        self.targets = df[TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        full_path = os.path.join(DATA_DIR, path)
        try:
            image = Image.open(full_path).convert('RGB')
        except:
             image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

def load_data():
    df = pd.read_csv(TRAIN_CSV)
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    return targets_wide

def train_oof():
    df = load_data()
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(df), len(TARGETS)))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n--- Fold {fold} ---")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_dataset = BiomassDataset(train_df, transform=train_transform)
        val_dataset = BiomassDataset(val_df, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Val
            model.eval()
            val_loss = 0
            fold_val_preds = []
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(DEVICE), targets.to(DEVICE)
                    outputs = model(images)
                    val_loss += criterion(outputs, targets).item() * images.size(0)
                    fold_val_preds.append(outputs.cpu().numpy())
            
            val_loss /= len(val_dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'effnet_fold{fold}.pth'))
                oof_preds[val_idx] = np.vstack(fold_val_preds)
                
            print(f"Fold {fold} Epoch {epoch+1} Val RMSE: {np.sqrt(val_loss):.4f}")

    # Save OOF
    oof_df = pd.DataFrame(oof_preds, columns=[f'OOF_EffNet_{t}' for t in TARGETS])
    oof_df['image_path'] = df['image_path']
    oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_effnet.csv'), index=False)
    
    total_rmse = np.sqrt(mean_squared_error(df[TARGETS], oof_preds))
    print(f"\nOverall EffNet OOF RMSE: {total_rmse:.4f}")

if __name__ == "__main__":
    train_oof()
