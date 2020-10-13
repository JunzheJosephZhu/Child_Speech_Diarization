import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, widths):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Conv1d(widths[i], widths[i + 1], 1))
        
    def forward(self, x, mask):
        '''
            x: [B, H, L]
            mask: [B, L], unused
            output: [B, O, L]
        '''
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class Pool_MLP(nn.Module):
    def __init__(self, widths):
        super(Pool_MLP, self).__init__()
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Conv1d(widths[i], widths[i + 1], 1))

    def forward(self, x, mask):
        '''
            x: [B, H, L]
            mask: [B, L]
            output: [B, O]
        '''
        assert mask.dtype == torch.bool and x.size(-1) == mask.size(-1)
        mask = mask.unsqueeze(1)
        x = x.masked_fill(mask, -np.inf)
        # [B, H, 1]
        x = self.pool(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x.squeeze(dim=2)

class MLP_Pool(nn.Module):
    def __init__(self, widths):
        super(MLP_Pool, self).__init__()
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Conv1d(widths[i], widths[i + 1], 1))

    def forward(self, x, mask):
        '''
            x: [B, H, L]
            mask: [B, L]
            output: [B, O]
        '''
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)

        assert mask.dtype == torch.bool and x.size(-1) == mask.size(-1)
        mask = mask.unsqueeze(1)
        x = x.masked_fill(mask, -np.inf)
        # [B, H, 1]
        x = self.pool(x)

        return x.squeeze(dim=2)

if __name__ == "__main__":
    embedding = torch.randn(2, 512, 80)
    mask = torch.ones(2, 80, dtype=torch.bool)
    mask[:, :40] = 0
    mlp = Pool_MLP([512, 3])
    print(mlp(embedding, mask))