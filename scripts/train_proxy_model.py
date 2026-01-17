
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
import joblib

# Configuration
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_EXAMPLE = os.path.join(DATA_DIR, 'test.csv') # Actually just a list of images usually
OUTPUT_DIR = 'models/proxy_model_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# The targets for THIS model are the metadata we want to impute
PROXY_TARGETS = ['Height_Ave_cm', 'Pre_GSHH_NDVI']

class ProxyMetadataDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.image_paths = df['image_path'].values
        
        if not self.is_test:
            self.targets = df[PROXY_TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_path = os.path.join(DATA_DIR, self.image_paths[idx])
        # Handle cases where path might be just filename or relative
        if not os.path.exists(img_path):
             # Try checking if it's inside data dir directly? 
             pass

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224)) # Should not happen in normally setup data
            
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.image_paths[idx]
        
        # Targets
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

class ProxyPredictor(nn.Module):
    def __init__(self):
        super(ProxyPredictor, self).__init__()
        # Use a solid backbone
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final layer
        in_features = self.cnn.fc.in_features
        
        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.2), # Evaluate if this helps
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, len(PROXY_TARGETS)) # Output: [Height, NDVI]
        )
        
    def forward(self, x):
        return self.cnn(x)

def load_data():
    print("Loading data for Proxy Model...")
    df = pd.read_csv(TRAIN_CSV)
    
    # We only need one row per image. The original train.csv has 5 rows per image (one per target).
    # We can group by image_path and take the first of Height/NDVI (since they are constant for the image)
    
    unique_images_df = df.groupby('image_path')[PROXY_TARGETS].first().reset_index()
    
    # Drop any that might still be missing (though report said none)
    unique_images_df = unique_images_df.dropna(subset=PROXY_TARGETS)
    
    return unique_images_df

def train_model():
    df = load_data()
    print(f"Unique Images Found: {len(df)}")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    train_dataset = ProxyMetadataDataset(train_df, transform=train_transform)
    val_dataset = ProxyMetadataDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = ProxyPredictor().to(DEVICE)
    
    criterion = nn.MSELoss() # Standard regression loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    
    print("\nStarting Proxy Model Training (Image -> Height/NDVI)...")
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
            
        scheduler.step()
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
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_rmse = np.sqrt(val_epoch_loss)
        
        # Calculate R2 for logging
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        r2_height = r2_score(all_targets[:,0], all_preds[:,0])
        r2_ndvi = r2_score(all_targets[:,1], all_preds[:,1])
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.4f} | R2 Height: {r2_height:.2f} | R2 NDVI: {r2_ndvi:.2f}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'proxy_best.pth'))
            
    print(f"\nTraining Complete. Best Val MSE: {best_val_loss:.4f}")
    
    # ---- INFERENCE ----
    print("\nGenerating Proxy Predictions for Test Set...")
    # Load test set (list of images) - we assume test_images/ folder or from a csv?
    # Inspecting report.md, it says "provided test.csv contained only image files"
    
    if os.path.exists('csiro-biomass/test.csv'):
        test_df = pd.read_csv('csiro-biomass/test.csv')
        # Ensure it has 'image_path' column? Or convert ID to path?
        # Assuming format similar to train
        if 'image_path' not in test_df.columns:
            # Maybe it has just ID?
            # Let's inspect test.csv structure in next steps if this fails, but for now write defensive code
            pass 
        
        dataset_test = ProxyMetadataDataset(test_df, transform=val_transform, is_test=True)
        loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
        
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'proxy_best.pth')))
        model.eval()
        
        proxy_results = []
        
        with torch.no_grad():
            for images, paths in loader_test:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.cpu().numpy()
                
                for path, pred in zip(paths, preds):
                    proxy_results.append({
                        'image_path': path,
                        'Predicted_Height': pred[0],
                        'Predicted_NDVI': pred[1]
                    })
        
        proxy_df = pd.DataFrame(proxy_results)
        proxy_csv_path = os.path.join(OUTPUT_DIR, 'proxy_test_metadata.csv')
        proxy_df.to_csv(proxy_csv_path, index=False)
        print(f"Saved proxy metadata to {proxy_csv_path}")

if __name__ == "__main__":
    train_model()
