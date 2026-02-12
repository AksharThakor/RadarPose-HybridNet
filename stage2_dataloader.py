import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

class ActivityRecognitionDataset(Dataset):
    def __init__(self, df, precomputed_dir, target_len=30, train=True):
        self.df = df
        self.precomputed_dir = precomputed_dir
        self.target_len = target_len
        self.train = train
        self.label_map = {act: i for i, act in enumerate(sorted(self.df['Action'].unique()))}

    def _apply_stable_normalization(self, skeletons):
        # Root-Centric Translation
        origin = skeletons[0, 0, :].copy()
        skeletons = skeletons - origin

        # Global View Alignment (Anchor to first 5 frames hip vector)
        hip_r = np.mean(skeletons[:5, 1, :], axis=0)
        hip_l = np.mean(skeletons[:5, 4, :], axis=0)
        hip_vec = hip_l - hip_r
        hip_vec[1] = 0 
        
        angle = np.arctan2(hip_vec[2], hip_vec[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        
        skeletons = np.matmul(skeletons, R.T)

        # Global Scaling
        torso_vecs = skeletons[:, 8, :] - skeletons[:, 0, :]
        mean_torso_len = np.mean(np.linalg.norm(torso_vecs, axis=1))
        skeletons = skeletons / (mean_torso_len + 1e-6)
        return skeletons

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = f"{row['Subject']}_{row['Action']}_start{row['Start']}_end{row['End']}.npy"
        skeletons = np.load(os.path.join(self.precomputed_dir, file_name))
        
        # Resample -> Normalize
        orig_t = skeletons.shape[0]
        f = interp1d(np.linspace(0, 1, orig_t), skeletons.reshape(orig_t, -1), axis=0, kind='linear')
        resampled = f(np.linspace(0, 1, self.target_len)).reshape(self.target_len, 17, 3)
        
        normalized = self._apply_stable_normalization(resampled)
        
        # Training Augmentation: Small Gaussian Noise
        if self.train:
            normalized += np.random.normal(0, 0.005, normalized.shape)
            
        return torch.FloatTensor(normalized).permute(2, 0, 1), self.label_map[row['Action']]

    def __len__(self):
        return len(self.df)