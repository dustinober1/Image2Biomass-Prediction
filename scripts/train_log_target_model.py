
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
TEST_CSV_EXAMPLE = os.path.join(DATA_DIR, 'test.csv') # Only for image list
OUTPUT_DIR = 'models/exp3_log_space'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

class LogSpaceBiomassDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.image_paths = df['image_path'].values
        
        if not self.is_test:
            # IMPORTANT: Apply log1p to targets
            raw_targets = df[TARGETS].values.astype(np.float32)
            self.targets = np.log1p(raw_targets) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.image_paths[idx]
            
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

def load_data():
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV)
    targets_wide = df.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()
    full_df = targets_wide.dropna(subset=TARGETS)
    return full_df

def train_model():
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
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
    
    train_dataset = LogSpaceBiomassDataset(train_df, transform=train_transform)
    val_dataset = LogSpaceBiomassDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss() # Applying MSE on Log Targets
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_rmse = float('inf')
    
    print("\nStarting Log-Space Training...")
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
        
        # Validation on ORIGINAL SCALE
        model.eval()
        all_preds_log = []
        all_targets_log = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(images)
                
                all_preds_log.append(outputs.cpu().numpy())
                all_targets_log.append(targets.cpu().numpy())
                
        all_preds_log = np.vstack(all_preds_log)
        all_targets_log = np.vstack(all_targets_log)
        
        # Invert Log
        all_preds = np.expm1(all_preds_log)
        all_targets = np.expm1(all_targets_log) # Should match original values
        
        # Compute RMSE
        rmses = []
        for i in range(len(TARGETS)):
            rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Log Loss: {epoch_loss:.4f} | Val RMSE (Orig Scale): {avg_rmse:.4f}")
        
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'logspace_best.pth'))
            
    print(f"\nTraining Complete. Best Val RMSE: {best_val_rmse:.4f}")
    
    # ---- INFERENCE ----
    print("\nGenerating Predictions for Test Set...")
    if os.path.exists('csiro-biomass/test.csv'):
        # For submission, we need to iterate over test images
        # We can reuse the same loader strategy as previous scripts
        test_df = pd.read_csv('csiro-biomass/test.csv')
        
        # Ensure unique images
        if 'image_path' in test_df.columns:
            # Just create a DF with unique image paths to predict, then map back
             unique_test_images = test_df[['image_path']].drop_duplicates()
        else:
             # Fallback
             unique_test_images = test_df # Assuming it has image_path

        test_dataset = LogSpaceBiomassDataset(unique_test_images, transform=val_transform, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'logspace_best.pth')))
        model.eval()
        
        predictions_map = {} # path -> [pred_array]
        
        with torch.no_grad():
            for images, paths in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds_log = outputs.cpu().numpy()
                preds = np.expm1(preds_log)
                
                for path, pred in zip(paths, preds):
                    predictions_map[path] = pred
                    
        # Create Submission
        submission = []
        # Iterate over original test.csv rows to ensure correct order/IDs if needed
        # Or just construct from map if test.csv is just list of queries
        
        # Based on previous pattern:
        # ID1001187975__Dry_Clover_g, test/ID...
        
        # We need to output rows for each target
        for idx, row in test_df.iterrows():
            img_path = row['image_path']
            # sample_id = row['sample_id'] # If available
            # If standard submission format is required
            
            # The test.csv provided has: sample_id, image_path, target_name
            # So we iterate and fill
            target_name = row['target_name']
            sample_id = row['sample_id']
            
            if img_path in predictions_map:
                pred_vec = predictions_map[img_path]
                # Find index of target_name
                try:
                    t_idx = TARGETS.index(target_name)
                    val = max(0, pred_vec[t_idx])
                except ValueError:
                    val = 0 # Should not happen
                
                submission.append({'sample_id': sample_id, 'target': val})
            else:
                submission.append({'sample_id': sample_id, 'target': 0})
                
        sub_df = pd.DataFrame(submission)
        out_file = os.path.join(OUTPUT_DIR, 'submission_exp3.csv')
        sub_df.to_csv(out_file, index=False)
        print(f"Saved submission to {out_file}")

if __name__ == "__main__":
    train_model()
