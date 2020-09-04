import torch
class Conv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, mp=1, dropout=0.0):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = kernel_size//2)
        self.batchnorm = torch.nn.BatchNorm1d(in_channels)
        self.act = torch.nn.ReLU()
        self.bn = bn
        if mp>1:
            self.mp = torch.nn.MaxPool1d(mp, padding=mp//2)
        else:
            self.mp = None
        self.dropout = nn.Dropout(dropout) if dropout else None
    def forward(self, X):
        if self.dropout:
            X = self.dropout(X)
        if self.bn:
            X = self.batchnorm(X)
        if self.mp:
            X = self.mp(X)
        X = self.act(self.conv(X))
        return X

class TemporalConv(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.modulelist = torch.nn.ModuleList()
        for layer in layers:
            self.modulelist.append(Conv1DBlock(*layer))
    def forward(self, x):
        """
            x: raw audio, [B, T]
        """
        # [B, 1, T]
        x = x.unsqueeze(1)
        for module in self.modulelist:
            x = module(x)
        return x
if __name__ == "__main__":
    num_spks = 4
    # in_channels, out_channels, kernel_size, stride, bn, mp, dropout
    layers = [(1, 80, 21, 3, False),
            (80, 160, 7, 1, True, 3),
            (160, 160, 3),
            (160, 160, 3),
            (160, 160, 3, 3),
            (160, 160, 3, 1, True, 3),
            (160, 160, 3, 3),
            (160, 160, 3, 1, True, 3),
            (160, 160, 13),
            (160, 2048, 15),
            (2048, 2048, 1),
            (2048, 2048, 1, 1, False),
            (2048, num_spks, 1, 1, False)]

    net = TemporalConv(layers).cuda()
    test_input = torch.rand(1, 9**3*2+1).cuda()
    output = net(test_input)
    print(output.shape)