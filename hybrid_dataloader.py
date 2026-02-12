import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from stage1_dataloader import MMFi_mmWave_Dataset 

class HybridMMWaveDataset(Dataset):
    def __init__(self, df, data_root, aggregation_size=5):
        self.df = df
        self.data_root = data_root
        # Reuse your existing aggregation logic
        self.loader_helper = MMFi_mmWave_Dataset(data_root, subjects=[], actions=[])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mmwave_dir = os.path.join(self.data_root, row['Environment'], row['Subject'], row['Action'], 'mmwave')
        
        frames = []
        for f_idx in range(row['Start'] - 1, row['End']):
            # Aggregation (0.5s window) as defined in Stage 1
            points = self.loader_helper._get_aggregated_points(mmwave_dir, f_idx)
            
            # Pad/Clip points to fixed count for batching (e.g., 128 points)
            if len(points) > 128:
                points = points[:128]
            elif len(points) < 128:
                pad = np.zeros((128 - len(points), 5))
                points = np.vstack([points, pad])
            frames.append(points)
            
        # Segment shape: [Time, Points=128, Channels=5]
        return torch.FloatTensor(np.array(frames)), row['Action']

def hybrid_collate(batch):
    # Padding segments to the longest length in the batch (up to 30)
    data, labels = zip(*batch)
    max_t = max([d.shape[0] for d in data])
    
    padded_data = []
    for d in data:
        t, p, c = d.shape
        pad = torch.zeros((max_t - t, p, c))
        padded_data.append(torch.cat([d, pad], dim=0))
        
    return torch.stack(padded_data), labels