
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/exp10_multitask'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BIOMASS_TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
AUX_TARGETS = ['Height_Ave_cm', 'Pre_GSHH_NDVI']

class MultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = df['image_path'].values
        self.biomass_targets = df[BIOMASS_TARGETS].values.astype(np.float32)
        self.aux_targets = df[AUX_TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        biomass = torch.tensor(self.biomass_targets[idx], dtype=torch.float32)
        aux = torch.tensor(self.aux_targets[idx], dtype=torch.float32)
        return image, biomass, aux

class MultiTaskResNet(nn.Module):
    def __init__(self, num_biomass=5, num_aux=2):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove original FC
        
        self.biomass_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_biomass)
        )
        
        self.aux_head = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
            nn.Linear(64, num_aux)
        )

    def forward(self, x):
        features = self.backbone(x)
        biomass_out = self.biomass_head(features)
        aux_out = self.aux_head(features)
        return biomass_out, aux_out

def load_data():
    df = pd.read_csv(TRAIN_CSV)
    
    # Pivot targets
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Extract aux targets
    aux_df = df.groupby('image_path')[AUX_TARGETS].first().reset_index()
    
    full_df = pd.merge(targets_wide, aux_df, on='image_path')
    full_df = full_df.dropna(subset=BIOMASS_TARGETS + AUX_TARGETS)
    print(f"Loaded {len(full_df)} samples")
    return full_df

def train_exp10():
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = MultiTaskDataset(train_df, train_transform)
    val_ds = MultiTaskDataset(val_df, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MultiTaskResNet().to(DEVICE)
    
    # Loss: Weighted sum of biomass MSE and aux MSE
    # Auxiliary targets help guide the backbone
    criterion_biomass = nn.HuberLoss() # More robust than MSE
    criterion_aux = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_rmse = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, biomass, aux in train_loader:
            images, biomass, aux = images.to(DEVICE), biomass.to(DEVICE), aux.to(DEVICE)
            
            optimizer.zero_grad()
            b_out, a_out = model(images)
            
            loss_b = criterion_biomass(b_out, biomass)
            loss_a = criterion_aux(a_out, aux)
            
            # Aux targets have different scales, but predicting Height (0-100) and NDVI (0-1)
            # is useful. We weight aux lower so main task remains primary.
            loss = loss_b + 0.1 * loss_a
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        
        # Eval
        model.eval()
        val_b_preds = []
        val_b_targets = []
        with torch.no_grad():
            for images, biomass, aux in val_loader:
                images, biomass = images.to(DEVICE), biomass.to(DEVICE)
                b_out, _ = model(images)
                val_b_preds.append(b_out.cpu().numpy())
                val_b_targets.append(biomass.cpu().numpy())
                
        val_b_preds = np.vstack(val_b_preds)
        val_b_targets = np.vstack(val_b_targets)
        
        rmses = []
        for i in range(len(BIOMASS_TARGETS)):
            rmse = np.sqrt(mean_squared_error(val_b_targets[:, i], val_b_preds[:, i]))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_ds):.4f} | Val RMSE: {avg_rmse:.4f}")
        
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'multitask_best.pth'))
            print("  New Best Model!")

    print(f"\nFinal Best Multi-Task Val RMSE: {best_val_rmse:.4f}")

if __name__ == "__main__":
    train_exp10()
