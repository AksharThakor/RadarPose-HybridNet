import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import hybrid components
from hybrid_model import MMWaveHybridE2E
from hybrid_dataloader import HybridMMWaveDataset, hybrid_collate

# Configuration
CONFIG = {
    "csv_path": "./data/MMFi_action_segments_rmA1_2_3_6_len10to30.csv",
    "data_root": "./data/content/MMFi",
    "model_path": "best_hybrid_e2e_model.pth",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def run_evaluation():
    
    # Prepare Data
    df = pd.read_csv(CONFIG['csv_path'])
    _, val_df = train_test_split(
        df, test_size=0.20, stratify=df['Action'], random_state=42
    )
    
    actions = sorted(df['Action'].unique())
    label_to_id = {act: i for i, act in enumerate(actions)}
    id_to_label = {i: act for act, i in label_to_id.items()}

    val_set = HybridMMWaveDataset(val_df, CONFIG['data_root'])
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=4, collate_fn=hybrid_collate)

    # Load Model
    model = MMWaveHybridE2E(num_classes=len(actions)).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print(f"Successfully loaded hybrid model from {CONFIG['model_path']}")
    except FileNotFoundError:
        print(f"Error: Checkpoint {CONFIG['model_path']} not found.")
        return

    model.eval()

    # Evaluation Loop
    all_preds = []
    all_labels = []
    
    print(f"Evaluating {len(val_df)} segments...")
    with torch.no_grad():
        for x, labels in tqdm(val_loader):
            x = x.to(CONFIG['device'])
            label_ids = [label_to_id[l] for l in labels]
            
            # Hybrid model forward
            _, logits = model(x)
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_ids)

    # Results Reporting
    print("\n" + "="*65)
    print("HYBRID E2E CLASSIFICATION REPORT")
    print("="*65)
    print(classification_report(all_labels, all_preds, target_names=actions))
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Overall Macro F1-Score: {macro_f1:.4f}")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 14))
    
    # Normalize by row (recall) to see where each class is leaking
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=actions, yticklabels=actions)
    
    plt.title('Hybrid MM-Wave Action Recognition: Confusion Matrix (Recall)')
    plt.xlabel('Predicted Action')
    plt.ylabel('Ground Truth Action')
    plt.savefig('hybrid_confusion_matrix.png', bbox_inches='tight', dpi=300)
    print("\nConfusion matrix saved as 'hybrid_confusion_matrix.png'")
    plt.show()

    print("\n" + "="*65)
    print("TOP 3 CONFUSED PAIRS")
    print("="*65)
    
    np.fill_diagonal(cm_norm, 0)
    flat_indices = np.argsort(cm_norm.ravel())[-3:][::-1]
    for idx in flat_indices:
        gt_idx, pred_idx = np.unravel_index(idx, cm_norm.shape)
        val = cm_norm[gt_idx, pred_idx]
        print(f"{actions[gt_idx]} misclassified as {actions[pred_idx]}: {val*100:.1f}% of the time")

if __name__ == "__main__":
    run_evaluation()