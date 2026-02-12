import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)
        
        # Position encoding as suggested in Point Transformer paper
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos):
        # x: [B, N, C], pos: [B, N, 3]
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        # Position encoding
        p_rel = self.pos_mlp(pos)
        
        # Vector self-attention implementation
        # This approximates the Point Transformer block for regression
        attn = self.attn_mlp(q - k + p_rel)
        attn = F.softmax(attn, dim=1)
        
        res = attn * (v + p_rel)
        return res

class MMWavePoseTransformer(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        # Input features: x, y, z, doppler, intensity (5 channels)
        self.fc_in = nn.Linear(5, 64)
        
        # Feature extraction layers
        self.block1 = PointTransformerLayer(64, 128)
        self.block2 = PointTransformerLayer(128, 256)
        
        # Global pooling to get a fixed-size feature vector from variable points
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head for 17 joints * 3 coordinates
        self.regressor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3)
        )

    def forward(self, x):
        # x shape: [Batch, Points, 5]
        xyz = x[:, :, :3]
        
        feat = self.fc_in(x)
        feat = self.block1(feat, xyz)
        feat = self.block2(feat, xyz)
        
        # Pool across the "Points" dimension
        feat = feat.transpose(1, 2) # [B, C, N]
        feat = self.global_pool(feat).squeeze(-1) # [B, C]
        
        output = self.regressor(feat)
        return output.view(-1, 17, 3) # Final shape: [Batch, 17, 3]