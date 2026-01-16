
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
IMG_DIR = DATA_DIR  # Images paths in CSV are like 'train/...' or just relative?
                    # EDA showed: train/ID1011485656.jpg. And data dir is csiro-biomass.
                    # So os.path.join(DATA_DIR, row['image_path']) should work.
OUTPUT_DIR = 'models/image_baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # We need to ensure we have one row per image with all targets
        # The input df should already be pivoted/merged
        self.image_paths = df['image_path'].values
        self.targets = df[TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback for robustness
            print(f"Warning: Image not found {img_path}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

def load_data():
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV)
    
    # Pivot to wide format
    targets_long = df[['sample_id', 'target_name', 'target', 'image_path']]
    
    # Pivot to wide format using image_path as index
    # targets_wide = df.pivot(index='sample_id', columns='target_name', values='target').reset_index() # INCORRECT
    
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # Check shape
    print(f"Pivoted shape: {targets_wide.shape}")  # Should be close to 357
    
    # Drop rows with NaNs (now only truly missing images will be dropped)
    len_before = len(targets_wide)
    full_df = targets_wide.dropna(subset=TARGETS)
    print(f"Data shape: {full_df.shape} (dropped {len_before - len(full_df)} rows with NaNs)")
    
    return full_df

def train_model():
    df = load_data()
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Transforms
    # Images are 2000x1000. Resize to 224x224 (distorted) for standard ResNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BiomassDataset(train_df, transform=train_transform)
    val_dataset = BiomassDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 avoids multiprocessing issues on Mac sometimes
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    print("Initializing ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5) # 5 outputs
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_rmse = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(images)
                val_loss += criterion(outputs, targets).item() * images.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Calculate RMSE per target
        rmses = []
        for i, target_name in enumerate(TARGETS):
            rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
            rmses.append(rmse)
            
        avg_rmse = np.mean(rmses)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val RMSE: {avg_rmse:.4f}")
        
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'resnet18_best.pth'))
            print("  New best model saved!")

    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    
    # Granular Metrics for Best Model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'resnet18_best.pth')))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print("\n--- Image Baseline Performance ---")
    for i, target_name in enumerate(TARGETS):
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        print(f"{target_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    train_model()
