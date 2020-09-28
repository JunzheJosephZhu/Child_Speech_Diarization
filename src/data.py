from praatio import tgio
from scipy.io.wavfile import read
import numpy as np
import glob
import json
import torch
from collections import defaultdict
def _count_frames(data_len, shift):
    # HACK: Assuming librosa.stft(..., center=True)
    n_frames = 1 + int(data_len / shift)
    if data_len % shift == 0:
        n_frames = n_frames - 1
    return n_frames

class LENADataSet(torch.utils.data.Dataset):
    def __init__(self, scp_file, chunk_size, stride, sr, maxmin, target_channels, spkr2idx):
        '''
            data_dir: scp file
            chunk_size: in frames
            stride: in samples
            both modes of encoder should have padding
        '''
        super().__init__()
        with open(scp_file) as file:
            self.data = json.load(file)
        sounds = []
        labels = []
        for wavfile in self.data:
            _sr, sound = read(wavfile)
            assert sr == _sr
            sound = np.float32(sound)
            num_frames = _count_frames(len(sound), stride)
            label = np.zeros((target_channels, num_frames), dtype = np.int32)
            if maxmin: # mimax norm
                _min, _max = np.amin(sound), np.amax(sound)
                sound = (sound - _min) / (_max - _min)
            else: # z norm
                sound = sound - sound.mean()
                sound /= sound.std()
            annofile = self.data[wavfile]
            tg = tgio.openTextgrid(annofile)
            for spkr, idx in spkr2idx.items():
                try:
                    entries = tg.tierDict[spkr].entryList
                except:
                #    print('no entry', spkr)
                    continue
                for entry in entries: # create label
                    st = int(entry.start * sr / stride)
                    ed = int(entry.end * sr / stride)
                    if ed > num_frames:
                        ed = num_frames
                    label[st:ed] = idx # use slicing for faster writings
            start = 0 # in frames
            while start + chunk_size < num_frames:
                sounds.append(sound[start * stride : (start + chunk_size) * stride])
                labels.append(label[:, start : start + chunk_size])
                start += chunk_size
        self.sounds, self.labels = sounds, labels
    def __len__(self):
        return len(self.sounds)
    def __getitem__(self, idx):
        return self.sounds[idx], self.labels[idx]
if __name__ == '__main__':
    args =  {"sr": 16000,
            "chunk_size": 80,
            "stride": 4096,
            "spkr2idx" : {"CHN":0, "CXN":0, "FAN":1, "MAN":2},
            "target_channels": 3,
            "maxmin": 1}
    trainset = LENADataSet('/ws/ifp-10_3/hasegawa/junzhez2/LENA/train.scp', **args)
    testset = LENADataSet('/ws/ifp-10_3/hasegawa/junzhez2/LENA/test.scp', **args)
    sound, label = trainset[20]
    print(len(trainset), len(testset))
    print(sound.shape, label.shape)