import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from hybrid_model import MMWaveHybridE2E
from hybrid_dataloader import HybridMMWaveDataset, hybrid_collate

CONFIG = {
    "csv_path": "./data/MMFi_action_segments_rmA1_2_3_6_len10to30.csv",
    "data_root": "./data/content/MMFi",
    "backbone_weights": "best_mmwave_model.pth",
    "checkpoint_path": "best_hybrid_e2e_model.pth",
    "batch_size": 32,
    "epochs": 50,
    "lr_classifier": 1e-4,
    "lr_backbone": 1e-6,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train():
    df = pd.read_csv(CONFIG['csv_path'])
    train_df, val_df = train_test_split(df, test_size=0.20, stratify=df['Action'], random_state=42)
    
    actions = sorted(df['Action'].unique())
    label_to_id = {act: i for i, act in enumerate(actions)}
    
    train_loader = DataLoader(HybridMMWaveDataset(train_df, CONFIG['data_root']), 
                              batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=4, collate_fn=hybrid_collate, pin_memory=True)
    val_loader = DataLoader(HybridMMWaveDataset(val_df, CONFIG['data_root']), 
                            batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=4, collate_fn=hybrid_collate, pin_memory=True)

    model = MMWaveHybridE2E(num_classes=len(actions), backbone_weights=CONFIG['backbone_weights']).to(CONFIG['device'])
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': model.classifier.parameters(), 'lr': CONFIG['lr_classifier']}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for x, labels in pbar:
            x = x.to(CONFIG['device'])
            label_ids = torch.tensor([label_to_id[l] for l in labels]).to(CONFIG['device'])
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                _, logits = model(x)
                loss = criterion(logits, label_ids)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = logits.max(1)
            train_total += label_ids.size(0)
            train_correct += predicted.eq(label_ids).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*train_correct/train_total:.2f}%"})

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(CONFIG['device'])
                label_ids = torch.tensor([label_to_id[l] for l in labels]).to(CONFIG['device'])
                with torch.amp.autocast('cuda'):
                    _, logits = model(x)
                _, predicted = logits.max(1)
                val_total += label_ids.size(0)
                val_correct += predicted.eq(label_ids).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print("--> Saved Best Hybrid Model!")

if __name__ == "__main__":
    train()