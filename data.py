from praatio import tgio
from scipy.io.wavfile import read
import numpy as np
import glob
import json
import torch

def update_mask(mask, i, ignore_len):
    lower = i-ignore_len//2
    upper = i+ignore_len//2
    mask[lower:upper] = 0.0

class LENADataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, ignore_duration = 0.0, segment_len = 32, stride = 16, sr = 16000, minmax = True):
        super().__init__()
        with open(data_dir) as file:
            self.data = json.load(file)
        sounds = []
        labels = []
        masks = []
        for wavfile in self.data:
            spkr2idx = {'CHN':1, 'CXN':1, 'FAN':1, 'MAN':1}
            _sr, sound = read(wavfile)
            assert sr == _sr
            sound = np.int32(sound)
            label = np.zeros(sound.shape, dtype = np.int64) # same shape as sound
            mask = np.ones(sound.shape, dtype = np.float32) # same shape as sound, default 1
            if minmax:
                _min, _max = np.amin(sound), np.amax(sound)
                sound = (sound - _min)/(_max - _min)
            else:
                sound = sound - sound.mean() # unit normalization
                sound /= sound.std()
            annofile = self.data[wavfile]
            tg = tgio.openTextgrid(annofile)
            for spkr, idx in spkr2idx.items(): # for each tier
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
                    update_mask(mask, start, int(sr*ignore_duration)) # write zeros to transition regions of mask
                    update_mask(mask, end, int(sr*ignore_duration)) # write zeros to transition regions of mask
            start = 0
            while start + segment_len * sr < len(sound):
                sounds.append(sound[start: start + segment_len * sr])
                labels.append(label[start: start + segment_len * sr])
                masks.append(mask[start: start + segment_len * sr])
                start += stride * sr
        self.sounds, self.labels, self.masks = sounds, labels, masks
    def __len__(self):
        return len(self.sounds)
    def __getitem__(self, idx):
        return self.sounds[idx], self.labels[idx], self.masks[idx]

if __name__ == '__main__':
    trainset = LENADataSet('/home/joseph/Desktop/LENA_praat/train.scp')
    testset = LENADataSet('/home/joseph/Desktop/LENA_praat/test.scp')
    sound, label, mask = trainset[0]
    print(len(trainset), len(testset))