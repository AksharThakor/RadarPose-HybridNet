import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class CTR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(CTR_Block, self).__init__()
        self.inter_channels = out_channels // 4
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        # Reduced space for topology refinement
        self.conv1 = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # Full space for feature projection
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        
        # Compute Topology Refinement Map (a)
        # x1, x2 shape: [N, inter_channels, V]
        x1 = self.conv1(x).mean(-2) 
        x2 = self.conv2(x).mean(-2)
        
        # a shape: [N, inter_channels, V, V]
        a = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        a = torch.softmax(a, dim=-1)
        
        # Reshape features to match refinement map channels
        # Split out_channels (64) into 4 groups of inter_channels (16)
        # New shape: [N, 4, 16, T, V]
        x3 = self.conv3(x).view(N, 4, self.inter_channels, T, V)
        
        # Apply Refined Topology via Einstein Summation
        # 'ncuv' is refinement map 'a' [N, 16, V, V]
        # 'nmctv' is feature map 'x3' [N, 4, 16, T, V]
        y = torch.einsum('ncuv,nmctv->nmctu', a, x3)
        y = y.reshape(N, self.out_channels, T, V)
        
        # Combine with the static Physical Adjacency Matrix
        y = y + torch.einsum('uv,nctv->nctu', self.A, self.conv4(x))
        
        y = self.tcn(y) + self.residual(x)
        return self.relu(y)

class CTRGCN(nn.Module):
    def __init__(self, num_classes=23, in_channels=3, num_point=17):
        super(CTRGCN, self).__init__()
        #Define Graph
        edges = [(10,9),(9,8),(8,7),(7,0),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16),(0,4),(4,5),(5,6),(0,1),(1,2),(2,3)]
        A = np.zeros((num_point, num_point))
        for i, j in edges:
            A[i, j] = A[j, i] = 1
        for i in range(num_point): A[i, i] = 1
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        self.A = D @ A @ D

        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        
        self.l1 = CTR_Block(in_channels, 64, self.A)
        self.l2 = CTR_Block(64, 64, self.A)
        self.l3 = CTR_Block(64, 128, self.A, stride=2)
        self.l4 = CTR_Block(128, 128, self.A)
        self.l5 = CTR_Block(128, 256, self.A, stride=2)
        self.l6 = CTR_Block(256, 256, self.A)
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        #x: [N, C, T, V]
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        return self.fc(x)