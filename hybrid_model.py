import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1_model import MMWavePoseTransformer
from stage2_model import CTRGCN

class MMWaveHybridE2E(nn.Module):
    def __init__(self, num_classes=23, backbone_weights=None):
        super().__init__()
        self.backbone = MMWavePoseTransformer(num_joints=17)
        if backbone_weights:
            self.backbone.load_state_dict(torch.load(backbone_weights))
            print("Loaded pretrained Stage 1 weights.")
        
        self.classifier = CTRGCN(num_classes=num_classes, in_channels=3, num_point=17)

    def forward(self, x):
        B, T_orig, P, C = x.shape
        
        # Point Transformer Backbone
        x_flat = x.view(B * T_orig, P, C)
        skeletons = self.backbone(x_flat) 
        skeletons = skeletons.view(B, T_orig, 17, 3) 
        
        # Differentiable Temporal Resampling
        skel_resample = skeletons.view(B, T_orig, -1).permute(0, 2, 1) 
        skel_resample = F.interpolate(skel_resample, size=30, mode='linear', align_corners=True)
        skeletons_30 = skel_resample.permute(0, 2, 1).view(B, 30, 17, 3)
        
        # Stable Global Normalization (Fixed Broadcasting)
        origin = skeletons_30[:, 0:1, 0:1, :] 
        skeletons_30 = skeletons_30 - origin
        
        # Calculate mean torso length across all 30 frames
        # Resulting shape: [B, 1]
        torso_len = torch.norm(skeletons_30[:, :, 8, :] - skeletons_30[:, :, 0, :], dim=-1).mean(dim=1, keepdim=True)
        
        # Reshape to [B, 1, 1, 1] for proper broadcasting with [B, 30, 17, 3]
        torso_len = torso_len.view(B, 1, 1, 1)
        skeletons_30 = skeletons_30 / (torso_len + 1e-6)
        
        # Classification
        gcn_input = skeletons_30.permute(0, 3, 1, 2).contiguous()
        logits = self.classifier(gcn_input)
        
        return skeletons_30, logits