import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # For Mixed Precision
import numpy as np
import random
from tqdm import tqdm
from stage1_dataloader import MMFi_mmWave_Dataset, collate_mmwave
from stage1_model import MMWavePoseTransformer


DATA_ROOT = "./data/content/MMFi"
BATCH_SIZE = 512  
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

def calculate_mpjpe(predicted, target):
    # Align root (Joint 0)
    root_pred = predicted[:, 0:1, :] 
    root_gt = target[:, 0:1, :]      
    pred_aligned = predicted - root_pred
    gt_aligned = target - root_gt
    
    dist = torch.norm(pred_aligned - gt_aligned, dim=2)
    return torch.mean(dist) * 1000 # Convert to mm

def train():
    # Dataset Setup (S1 Random Split)
    all_subjects = [f'S{i:02d}' for i in range(1, 41)]
    all_actions = [f'A{i:02d}' for i in range(1, 28)]
    all_pairs = [(s, a) for s in all_subjects for a in all_actions]
    random.seed(42)
    random.shuffle(all_pairs)
    
    split_idx = int(0.75 * len(all_pairs))
    train_dataset = MMFi_mmWave_Dataset(DATA_ROOT, subjects=[p[0] for p in all_pairs[:split_idx]], actions=all_actions)
    test_dataset = MMFi_mmWave_Dataset(DATA_ROOT, subjects=[p[0] for p in all_pairs[split_idx:]], actions=all_actions)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_mmwave, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=collate_mmwave, num_workers=4, pin_memory=True)

    
    model = MMWavePoseTransformer(num_joints=17).to(DEVICE)
    scaler = GradScaler() # Scaler for FP16 training
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01) # AdamW is often better for Transformers
    criterion = nn.MSELoss()
    
    best_mpjpe = float('inf')

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for points, gts in train_pbar:
            points, gts = points.to(DEVICE, non_blocking=True), gts.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            
            # Mixed Precision Forward Pass
            with autocast():
                outputs = model(points)
                loss = criterion(outputs, gts)
            
            # Scaler for mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Fast Validation
        model.eval()
        total_mpjpe = 0
        with torch.no_grad(), autocast():
            for points, gts in tqdm(test_loader, desc=f"Epoch {epoch+1} [Val]"):
                points, gts = points.to(DEVICE, non_blocking=True), gts.to(DEVICE, non_blocking=True)
                outputs = model(points)
                total_mpjpe += calculate_mpjpe(outputs, gts).item()
        
        avg_mpjpe = total_mpjpe / len(test_loader)
        print(f"\nEpoch {epoch+1} Results: Val MPJPE: {avg_mpjpe:.2f}mm")

        if avg_mpjpe < best_mpjpe:
            best_mpjpe = avg_mpjpe
            torch.save(model.state_dict(), "best_mmwave_model.pth")
            print(f"--> Saved best model ({best_mpjpe:.2f}mm)")

if __name__ == "__main__":
    train()