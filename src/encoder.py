import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

# 250 ms per frame
class AE(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(AE, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, input):
        o = input.unsqueeze(1)

        # Down Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)
        return o

# 125 * 2 = 250 ms per frame
class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=4001, hop_length=2048, n_mels=23, context_size=7, subsample=2):
        super(LogMel, self).__init__()

        self.stft = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.pad = nn.ReplicationPad1d(padding=context_size)
        self.context_size = context_size
        self.subsample = subsample

    def forward(self, waveform):
        mel_specgram = self.stft(waveform)
        log_offset = 1e-6
        logmel = torch.log(mel_specgram + log_offset)
        logmel = self.pad(logmel)
        spliced_logmel = logmel.unfold(dimension=-1, size=self.context_size * 2 + 1, step=1)
        # [B, feature_dim, T, context_size * 2 + 1]
        spliced_logmel = spliced_logmel[:, :, ::self.subsample] 
        # [B, context_size * 2 + 1, feature_dim, T]
        spliced_logmel = spliced_logmel.permute(0, 3, 1, 2).contiguous() 
        # [B, (context_size * 2 + 1) * feature_dim, T]
        B, _, _, T = spliced_logmel.size()
        spliced_logmel = spliced_logmel.view(B, -1, T)
        return spliced_logmel

if __name__ == "__main__":
    import os
    sae = AE()
    mel = LogMel()
    input = torch.randn(1, 4096 * 80)
    print(sae(input).shape)
    print(mel(input))