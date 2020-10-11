import torch
import torch.nn as nn
import torch.nn.functional as F

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=5):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, features, mask):
        '''
            features: [B, C, L], C=input_size
            output: [B, H, L], H=hidden_size
        '''
        # [L, B, C]
        features = features.permute(2, 0, 1)
        # [L, B, H]
        x, _ = self.lstm(features)
        # [B, H, L]
        x = x.permute(1, 2, 0)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff):
        super(EncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff1 = nn.Linear(hidden_size, ff)
        self.ff2 = nn.Linear(ff, hidden_size)

    def forward(self, x, mask):
        '''
            args:
                x: [L, B, H]
        '''
        x = self.ln1(x)
        skip = x
        x, _ = self.attention(x, x, x, key_padding_mask=mask) 
        x = x + skip
        x = self.ln2(x)
        skip = x
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)
        x = x + skip
        return x


class MHA(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_heads=4, num_layers=2, ff_size=1024):
        super(MHA, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.attention_modules = nn.ModuleList()
        for i in range(num_layers):
            self.attention_modules.append(EncoderBlock(hidden_size, num_heads, ff_size))
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, features, mask):
        '''
            featuers: [B, C, L]
            mask: [B, L]
            output: [B, H, L]
        '''
        assert mask.dtype == torch.bool
        # [L, B, C]
        features = features.permute(2, 0, 1)
        # [L, B, H]
        x = self.linear(features)
        for attention_module in self.attention_modules:
            x = attention_module(x, mask)
        x = self.ln(x)
        # [B, H, L]
        x = x.permute(1, 2, 0)
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 288, 80)
    rnn = BLSTM(288)
    mha = MHA(288)
    mask = torch.ones(2, 80, dtype=bool)
    mask[0, :40] = False
    print(rnn(x, mask).shape)
    print(mha(x, mask))