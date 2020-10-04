import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, widths):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(torch.nn.Conv1d(widths[i], widths[i + 1], 1))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
    
if __name__ == "__main__":
    embedding = torch.randn(2, 512, 80)
    mlp = MLP([512, 3])
    print(mlp(embedding).shape)