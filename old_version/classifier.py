import torch
from torch import nn
class Conv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, mp=1, dropout=0.0, use_relu=True, res=True):
        super().__init__()
        self.use_relu = use_relu
        self.bn = bn
        self.res = res
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride = stride) if in_channels != out_channels or stride!=1 else None
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride = stride, padding = kernel_size//2)
        self.batchnorm1 = torch.nn.BatchNorm1d(in_channels)
        self.act = torch.nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding = kernel_size//2)
        self.batchnorm2 = torch.nn.BatchNorm1d(out_channels)
        self.mp = torch.nn.MaxPool1d(mp, padding=mp//2) if mp else None
        self.dropout = nn.Dropout(dropout) if dropout else None
    def forward(self, X):
        if self.dropout:
            X = self.dropout(X)
        copy = X
        X = self.conv1(X)
        if self.bn:
            X = self.batchnorm1(X)
        if self.use_relu:
            X = self.act(X)
        X = self.conv2(X)
        if self.bn:
            X = self.batchnorm2(X)
        if self.res:
            if self.shortcut:
                X += self.shortcut(copy)
            else:
                X += copy
        if self.mp:
            X = self.mp(X)
        if self.use_relu:
            X = self.act(X)
        return X

class TemporalConv(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.modulelist = []
        for layer in layers:
            self.modulelist.append(Conv1DBlock(*layer))
        self.modulelist = nn.ModuleList(self.modulelist)
    def forward(self, x):
        """
            x: raw audio, [B, T]
        """
        # [B, 1, T]
        x = x.unsqueeze(1)
        for module in self.modulelist:
            x = module(x)
        return x
    @staticmethod
    def serialize(model, optimizer, epoch, tr_acc=None, cv_acc=None, val_no_impv=None, random_state=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_acc is not None:
            package['tr_acc'] = tr_acc
            package['cv_acc'] = cv_acc
            package['val_no_impv'] = val_no_impv
            package['random_state'] = random_state
        return package

if __name__ == "__main__":
    import sys
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    num_spks = 3
    # in_channels, out_channels, kernel_size, stride, bn, mp, dropout, relu
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
            (2048, num_spks, 1, 1, False, 1, 0.0, False)]
    torch.manual_seed(0)
    net = TemporalConv(layers).cuda(3)
    test_input = torch.rand(1, 9**3*10+1).cuda(3)
    output = net(test_input)
    print(output)