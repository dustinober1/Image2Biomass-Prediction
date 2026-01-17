
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/multimodal_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
METADATA_COLS = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State', 'Species']

class MultimodalBiomassDataset(Dataset):
    def __init__(self, df, feature_cols, transform=None):
        self.df = df
        self.feature_cols = feature_cols
        self.transform = transform
        
        self.image_paths = df['image_path'].values
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_path = os.path.join(DATA_DIR, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224)) # Fallback
            
        if self.transform:
            image = self.transform(image)
            
        # Tabular features
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Targets
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return image, features, target

class MultimodalBiomassPredictor(nn.Module):
    def __init__(self, num_tabular_features, num_targets=5):
        super(MultimodalBiomassPredictor, self).__init__()
        
        # Image Branch (ResNet18)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the classification head (fc)
        # ResNet18 final layer is usually 512 features before the FC
        self.cnn_out_features = self.cnn.fc.in_features 
        self.cnn.fc = nn.Identity() 
        
        # Tabular Branch (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.tabular_out_features = 32
        
        # Fusion Layer
        fusion_input_dim = self.cnn_out_features + self.tabular_out_features
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_targets)
        )
        
    def forward(self, image, tabular):
        # Image Branch
        img_features = self.cnn(image) # (Batch, 512)
        
        # Tabular Branch
        tab_features = self.tabular_mlp(tabular) # (Batch, 32)
        
        # Concatenate
        combined = torch.cat((img_features, tab_features), dim=1)
        
        # Fusion
        output = self.fusion_head(combined)
        return output

def load_data():
    print("Loading and merging data...")
    df = pd.read_csv(TRAIN_CSV)
    
    # 1. Pivot Targets
    targets_pivot = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    
    # 2. Get Metadata (using image_path)
    # Each image_path maps to one set of metadata
    meta_df = df.groupby('image_path')[METADATA_COLS].first().reset_index()
    
    # 3. Merge
    full_df = pd.merge(targets_pivot, meta_df, on='image_path')
    
    # Drop rows with NaNs in targets
    full_df = full_df.dropna(subset=TARGETS)
    
    # Encode Categoricals
    le_state = LabelEncoder()
    full_df['State_Encoded'] = le_state.fit_transform(full_df['State'])
    
    le_species = LabelEncoder()
    full_df['Species_Encoded'] = le_species.fit_transform(full_df['Species'])
    
    # Save encoders
    joblib.dump(le_state, os.path.join(OUTPUT_DIR, 'le_state.pkl'))
    joblib.dump(le_species, os.path.join(OUTPUT_DIR, 'le_species.pkl'))
    
    feature_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'State_Encoded', 'Species_Encoded']
    
    return full_df, feature_cols

def train_model():
    df, feature_cols = load_data()
    print(f"Data Loaded: {len(df)} samples")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Scale Features (Important for MLP)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Higher res than baseline
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = MultimodalBiomassDataset(train_df.copy(), feature_cols, transform=train_transform)
    val_dataset = MultimodalBiomassDataset(val_df.copy(), feature_cols, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model Setup
    model = MultimodalBiomassPredictor(num_tabular_features=len(feature_cols), num_targets=len(TARGETS))
    model = model.to(DEVICE)
    
    criterion = nn.SmoothL1Loss() # Robust regression loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_rmse = float('inf')
    
    print("\nStarting Multimodal Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, features, targets in train_loader:
            images = images.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, features, targets in val_loader:
                images = images.to(DEVICE)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(images, features)
                # MSE for checking
                loss_mse = nn.MSELoss()(outputs, targets)
                val_loss += loss_mse.item() * images.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset) # This is effectively MSE
        val_rmse = np.sqrt(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'multimodal_best.pth'))
            # print("  New best model!")
            
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    
    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'multimodal_best.pth')))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, features, targets in val_loader:
            images = images.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(images, features)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print("\n--- Multimodal Model Performance ---")
    metrics = []
    for i, target_name in enumerate(TARGETS):
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        print(f"{target_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        metrics.append({'rmse': rmse, 'mae': mae, 'r2': r2})
        
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    avg_r2 = np.mean([m['r2'] for m in metrics])
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    print(f"Average R2:   {avg_r2:.4f}")

if __name__ == "__main__":
    train_model()
