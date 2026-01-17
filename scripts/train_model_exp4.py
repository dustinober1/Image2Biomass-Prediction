
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/exp4_efficientnet_tta'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 50 # EfficientNets can overfit quickly on small data
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

print(f"Using device: {DEVICE}")

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.image_paths = df['image_path'].values
        
        if not self.is_test:
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
            
        if self.is_test:
            return image, path
            
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

def load_data():
    df = pd.read_csv(TRAIN_CSV)
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    full_df = targets_wide.dropna(subset=TARGETS)
    return full_df

def get_model():
    # EfficientNet B0 - pretrained
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)
    return model

def train_model():
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Added flip
        transforms.RandomRotation(30), # Increased rot
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BiomassDataset(train_df, transform=train_transform)
    val_dataset = BiomassDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = get_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    
    print("Starting EfficientNet Training...")
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
                
                # Standard Inf
                outputs = model(images)
                val_loss += criterion(outputs, targets).item() * images.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        val_loss /= len(val_dataset)
        rmse = np.sqrt(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val RMSE: {rmse:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'effnet_best.pth'))
    
    print(f"Best Val RMSE: {np.sqrt(best_loss):.4f}")
    
    # ---- TTA INFERENCE ----
    # For Exp 4, we evaluate if TTA helps
    print("\nRunning TTA Evaluation on Validation Set...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'effnet_best.pth')))
    model.eval()
    
    tta_preds = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            # TTA: Original, HFlip, VFlip
            img, target = val_dataset[i] # 3, 224, 224
            
            # Create augmented batch
            originals = img.unsqueeze(0).to(DEVICE)
            hflip = transforms.functional.hflip(originals)
            vflip = transforms.functional.vflip(originals)
            
            # Stack
            batch = torch.cat([originals, hflip, vflip], dim=0)
            
            # Predict
            outs = model(batch) # 3, 5
            
            # Average
            avg_pred = torch.mean(outs, dim=0).cpu().numpy()
            tta_preds.append(avg_pred)
            actuals.append(target.numpy())
            
    tta_preds = np.vstack(tta_preds)
    actuals = np.vstack(actuals)
    
    rmse_tta = np.sqrt(mean_squared_error(actuals, tta_preds))
    print(f"Validation RMSE with TTA (3x): {rmse_tta:.4f}")
            
    # Inference for Submission (using TTA)
    print("Generating Submission for Exp 4...")
    if os.path.exists(os.path.join(DATA_DIR, 'test.csv')):
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        unique_paths = test_df[['image_path']].drop_duplicates()
        
        # Test Data Loader (No shuffle)
        test_dataset = BiomassDataset(unique_paths, transform=val_transform, is_test=True)
        # We process 1 by 1 for simple TTA loop logic, or batched TTA
        
        # Simple Batched Loop
        predictions_map = {}
        
        for i in range(len(test_dataset)):
            img, path = test_dataset[i]
            
            with torch.no_grad():
                originals = img.unsqueeze(0).to(DEVICE)
                hflip = transforms.functional.hflip(originals)
                vflip = transforms.functional.vflip(originals)
                batch = torch.cat([originals, hflip, vflip], dim=0)
                outs = model(batch)
                avg_pred = torch.mean(outs, dim=0).cpu().numpy()
                predictions_map[path] = avg_pred
                
        submission = []
        for idx, row in test_df.iterrows():
            path = row['image_path']
            s_id = row['sample_id']
            t_name = row['target_name']
            
            if path in predictions_map:
                pred_vec = predictions_map[path]
                t_idx = TARGETS.index(t_name)
                val = max(0, pred_vec[t_idx])
                submission.append({'sample_id': s_id, 'target': val})
            else:
                submission.append({'sample_id': s_id, 'target': 0})
                
        sub_df = pd.DataFrame(submission)
        out_file = os.path.join(OUTPUT_DIR, 'submission_exp4.csv')
        sub_df.to_csv(out_file, index=False)
        print(f"Saved submission to {out_file}")

if __name__ == "__main__":
    train_model()
