from praatio import tgio
from scipy.io.wavfile import read
import numpy as np
import glob
import json
import torch

class LENADataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, segment_len, stride, sr, minmax, layers, device_ids, use_mask, spkr2idx):
        '''
            data_dir: scp file
            segment_len: in seconds
            stride: in seconds
        '''
        super().__init__()
        self.stride = self._compute_stride(layers)
        self.device = device_ids[0]
        with open(data_dir) as file:
            self.data = json.load(file)
        sounds = []
        labels = []
        masks = []
        for wavfile in self.data:
            _sr, sound = read(wavfile)
            assert sr == _sr
            sound = np.float32(sound)
            label = np.zeros(sound.shape, dtype = np.int32) # same shape as sound
            mask = np.zeros(sound.shape, dtype = np.int32) # same shape as sound, default 1
            if minmax: # mimax norm
                _min, _max = np.amin(sound), np.amax(sound)
                sound = (sound - _min)/(_max - _min)
            else: # z norm
                sound = sound - sound.mean()
                sound /= sound.std()
            annofile = self.data[wavfile]
            tg = tgio.openTextgrid(annofile)
            for spkr, idx in spkr2idx.items():
                try:
                    entries = tg.tierDict[spkr].entryList
                except:
    #                print('no entry', spkr)
                    continue
                for entry in entries: # create label
                    start = int(entry.start*sr)
                    end = int(entry.end*sr)
                    if end>len(sound):
                        end = len(sound)
                    label[start:end] = idx # use slicing for faster writings
                    mask[start:end] += 1
            mask = np.int32(mask == 1) # dont care about noise/overlap
            if not use_mask:
                mask = np.ones(mask.shape, dtype=int)
            start = 0
            while start + segment_len * sr < len(sound):
                sounds.append(sound[start: start + segment_len * sr])
                labels.append(label[start: start + segment_len * sr][::self.stride])
                masks.append(mask[start: start + segment_len * sr][::self.stride])
                start += stride * sr
        self.sounds, self.labels, self.masks = sounds, labels, masks
    def __len__(self):
        return len(self.sounds)
    def __getitem__(self, idx):
        return torch.Tensor(self.sounds[idx]).cuda(self.device).float(), \
            torch.Tensor(self.labels[idx]).cuda(self.device).long(), \
            torch.Tensor(self.masks[idx]).cuda(self.device).float()
    def _compute_stride(self, layers):
        stride = 1
        for layer in layers:
            if len(layer) >= 4:
                stride *= layer[3]
            if len(layer) >= 6:
                stride *= layer[5]
        return stride
if __name__ == '__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
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
    trainset = LENADataSet('/ws/ifp-10_3/hasegawa/junzhez2/LENA/train.scp', 32, 16, 16000, True, layers)
    testset = LENADataSet('/ws/ifp-10_3/hasegawa/junzhez2/LENA/test.scp', 32, 16, 16000, True, layers)
    sound, label, mask = trainset[20]
    print(len(trainset), len(testset))
    print(sound.shape, label.shape, mask[::9**3])