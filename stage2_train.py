import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from stage2_dataloader import ActivityRecognitionDataset
from stage2_model import CTRGCN

CONFIG = {
    "csv_path": "./data/MMFi_action_segments_rmA1_2_3_6_len10to30.csv",
    "precomputed_dir": "./data/precomputed_skeletons",
    "checkpoint_path": "best_ctrgcn_stratified.pth",
    "batch_size": 128,
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.05,
    "patience": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def get_stratified_split(csv_path):
    df = pd.read_csv(csv_path)
    # 80/20 split, maintaining the same percentage of each Action (A04-A27) in both sets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['Action']
    )
    return train_df, val_df

def calculate_weights(df):
    actions = np.array(sorted(df['Action'].unique()))
    y = df['Action'].values
    weights = compute_class_weight(class_weight='balanced', classes=actions, y=y)
    return torch.tensor(weights, dtype=torch.float)

def train():
    # Prepare Stratified Data
    train_df, val_df = get_stratified_split(CONFIG['csv_path'])
    class_weights = calculate_weights(train_df).to(CONFIG['device'])
    
    train_set = ActivityRecognitionDataset(train_df, CONFIG['precomputed_dir'], train=True)
    val_set = ActivityRecognitionDataset(val_df, CONFIG['precomputed_dir'], train=False)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # Model Initialization
    model = CTRGCN(num_classes=23, in_channels=3, num_point=17).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()
    
    best_acc = 0.0
    no_improve = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for skeletons, labels in pbar:
            skeletons, labels = skeletons.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                outputs = model(skeletons)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'acc': f"{100.*correct/total:.2f}%"})

        #Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for skeletons, labels in val_loader:
                skeletons, labels = skeletons.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(skeletons)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        print(f"Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= CONFIG['patience']:
            print("Early Stopping Triggered")
            break

if __name__ == "__main__":
    train()