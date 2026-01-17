
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Config - Distillation
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
EXP1_SUBMISSION = 'models/exp1_proxy_metadata/submission_exp1.csv'
OUTPUT_DIR = 'models/exp5_pseudo_label'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100 # Can train longer with more data
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

class DistilledDataset(Dataset):
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

def prepare_distillation_data():
    print("Preparing Data for Pseudo-Labeling...")
    
    # 1. Original Train Data
    train_df_orig = pd.read_csv(TRAIN_CSV)
    train_pivot = train_df_orig.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    train_pivot = train_pivot.dropna(subset=TARGETS)
    print(f"Original Train Samples: {len(train_pivot)}")
    
    # 2. Pseudo-Labeled Test Data (from Exp 1)
    if not os.path.exists(EXP1_SUBMISSION):
        print(f"Exp 1 Submission not found at {EXP1_SUBMISSION}. Cannot run Exp 5.")
        return None
        
    sub_df = pd.read_csv(EXP1_SUBMISSION)
    # Format: sample_id (ID__Target), target
    
    # We need to pivot this back to wide format: image_path -> targets
    # Helper to parse ID
    test_rows = []
    
    # Extract unique IDs from sample_id
    # sample_id looks like: ID1001187975__Dry_Clover_g
    sub_df['id_base'] = sub_df['sample_id'].apply(lambda x: x.split('__')[0])
    sub_df['target_name'] = sub_df['sample_id'].apply(lambda x: x.split('__')[1])
    
    test_pivot = sub_df.pivot(index='id_base', columns='target_name', values='target').reset_index()
    
    # Map id_base back to image_path.
    # We assume test images are 'test/{id_base}.jpg'
    # Check if test folder is 'test/'
    test_pivot['image_path'] = test_pivot['id_base'].apply(lambda x: f"test/{x}.jpg")
    
    print(f"Pseudo-Labeled Test Samples: {len(test_pivot)}")
    
    # 3. Combine
    combined_df = pd.concat([train_pivot, test_pivot], axis=0, ignore_index=True)
    print(f"Combined Training Set: {len(combined_df)}")
    
    return combined_df

def get_model():
    # EfficientNet B0
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)
    return model

def train_distilled_model():
    combined_df = prepare_distillation_data()
    if combined_df is None:
        return

    # Split (Stratified if possible, but random is okay for now)
    # We want to validate on ORIGINAL TRAIN data only to see real performance?
    # Or validate on a hold-out of original train.
    
    # Let's use a hold-out from the Original Train set for true validation
    # Use 'train/' prefix to filter
    orig_train_mask = combined_df['image_path'].str.startswith('train/')
    orig_train_df = combined_df[orig_train_mask]
    pseudo_test_df = combined_df[~orig_train_mask]
    
    train_subset, val_subset = train_test_split(orig_train_df, test_size=0.2, random_state=42)
    
    # Final Train = TrainSubset + PseudoTest
    final_train_df = pd.concat([train_subset, pseudo_test_df], axis=0)
    print(f"Final Train Size: {len(final_train_df)} (Orig: {len(train_subset)} + Pseudo: {len(pseudo_test_df)})")
    print(f"Validation Size: {len(val_subset)} (Pure Original)")
    
    # Transforms (Strong Aug)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DistilledDataset(final_train_df, transform=train_transform)
    val_dataset = DistilledDataset(val_subset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = get_model().to(DEVICE)
    criterion = nn.SmoothL1Loss() # Robust loss for distillation
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    
    print("Starting Distilled Training (Exp 5)...")
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
        val_loss = 0.0 # MSE for metric
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(images)
                val_loss += nn.MSELoss()(outputs, targets).item() * images.size(0)
                
        val_rmse = np.sqrt(val_loss / len(val_dataset))
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'distilled_best.pth'))
            
    print(f"Best Distilled RMSE: {best_loss:.4f}")
    
    # Inference on Test (Self-Consistency Check?)
    # Generating submission just to see if image model improved
    
if __name__ == "__main__":
    train_distilled_model()
