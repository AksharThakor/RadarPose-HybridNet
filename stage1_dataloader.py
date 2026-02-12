import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class MMFi_mmWave_Dataset(Dataset):
    def __init__(self, data_root, subjects, actions, aggregation_size=5):
        self.data_root = data_root
        self.subjects = subjects
        self.actions = actions
        self.aggregation_size = aggregation_size
        self.data_list = self._build_index()
        
    

    def _build_index(self):
        data_info = []
        for scene in sorted(os.listdir(self.data_root)):
            if not scene.startswith('E'): continue
            for sub in self.subjects:
                sub_path = os.path.join(self.data_root, scene, sub)
                if not os.path.exists(sub_path): continue
                for act in self.actions:
                    act_path = os.path.join(sub_path, act)
                    mmwave_path = os.path.join(act_path, 'mmwave')
                    gt_path = os.path.join(act_path, 'ground_truth.npy')
                    
                    if os.path.exists(mmwave_path) and os.path.exists(gt_path):
                        # Each activity has exactly 297 frames
                        for idx in range(297):
                            data_info.append({
                                'mmwave_dir': mmwave_path,
                                'gt_path': gt_path,
                                'frame_idx': idx
                            })
        return data_info

    def _load_bin(self, path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return np.empty((0, 5))
        # MM-Fi uses float64 for mmWave bin files
        data = np.fromfile(path, dtype=np.float64)
        return data.reshape(-1, 5)

    def _get_aggregated_points(self, mmwave_dir, frame_idx):
        # Specific sliding window logic: [idx-2, idx-1, idx, idx+1, idx+2]
        # Indices are 0-based; file names are frame001.bin to frame297.bin
        indices = [
            max(0, min(296, frame_idx - 2)),
            max(0, min(296, frame_idx - 1)),
            frame_idx,
            max(0, min(296, frame_idx + 1)),
            max(0, min(296, frame_idx + 2))
        ]
        
        # F1 -> F1, F1, F1, F2, F3
        # F2 -> F1, F1, F2, F3, F4 ...
        if frame_idx == 0:
            indices = [0, 0, 0, 1, 2]
        elif frame_idx == 1:
            indices = [0, 0, 1, 2, 3]

        all_points = []
        for i in indices:
            file_path = os.path.join(mmwave_dir, f"frame{i+1:03d}.bin")
            points = self._load_bin(file_path)
            all_points.append(points)
            
        aggregated = np.concatenate(all_points, axis=0)
        return aggregated
    
    def _spatial_filter(self, points):
    # ROI boundaries based on your Ground Truth Analysis
        x_min, x_max = -0.8, 0.8
        y_min, y_max = -1.2, 1.2
        z_min, z_max = 2.5, 4.5
    
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
           (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
           (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
        filtered_points = points[mask]
    
        # If filtering removes everything, return a single dummy point at the subject's 
        # expected center (3.3m) to avoid breaking the Transformer layers.
        if len(filtered_points) == 0:
            return np.array([[0.0, 0.0, 3.3, 0.0, 0.0]])
        
        return filtered_points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Ground Truth Skeleton
        gt = np.load(item['gt_path'])[item['frame_idx']]
        
        # Aggregated mmWave Points
        points = self._get_aggregated_points(item['mmwave_dir'], item['frame_idx'])
        

        if points.shape[0] == 0:
            points = np.zeros((1, 5)) # Handle empty frames gracefully by adding a dummy point


 
        return torch.FloatTensor(points), torch.FloatTensor(gt)

def collate_mmwave(batch):
    
    points, gts = zip(*batch)
    padded_points = torch.nn.utils.rnn.pad_sequence(points, batch_first=True)
    gts = torch.stack(gts)
    return padded_points, gts

