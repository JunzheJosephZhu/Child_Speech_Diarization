from praatio import tgio
from scipy.io.wavfile import read
import numpy as np
import glob
import json
import torch
from collections import defaultdict
import os

def _count_frames(data_len, shift):
    # HACK: Assuming librosa.stft(..., center=True)
    n_frames = 1 + int(data_len / shift)
    if data_len % shift == 0:
        n_frames = n_frames - 1
    return n_frames

class LENADataSet(torch.utils.data.Dataset):
    def __init__(self, scp_file, dataset_root, chunk_size, stride, sr, minmax, target_channels, spkr2idx, duration_thres=0.1):
        '''
            data_dir: scp file
            chunk_size: in frames
            stride: in samples
            duration_thres: discard all chunks with speech duration below this
            both modes of encoder should have padding
        '''
        super().__init__()
        with open(os.path.join(dataset_root, scp_file)) as file:
            self.data = json.load(file)
        sounds = []
        labels = []
        for wavfile in self.data:
            _sr, sound = read(wavfile)
            assert sr == _sr
            sound = np.float32(sound)
            num_frames = _count_frames(len(sound), stride)
            label = np.zeros((target_channels, num_frames), dtype = float)
            if minmax: # mimax norm
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
                    label[idx, st:ed] = 1 # use slicing for faster writings
            start = 0 # in frames
            while start + chunk_size < num_frames:
                if label[:, start : start + chunk_size].sum() > chunk_size * duration_thres:
                    sounds.append(sound[start * stride : (start + chunk_size) * stride])
                    labels.append(label[:, start : start + chunk_size])
                start += chunk_size
        self.sounds, self.labels = sounds, labels

    def __len__(self):
        return len(self.sounds)

    def __getitem__(self, idx):
        return self.sounds[idx], self.labels[idx]

if __name__ == '__main__':
    root = "~/Desktop/MaxMin_Pytorch"
    config_filename = "configs/SAE_RNN.json"
    with open(os.path.expanduser(os.path.join(root, config_filename))) as file:
        config = json.load(file)
    dataset_config = config["dataset"]
    trainset = LENADataSet(dataset_config["train"], **dataset_config["args"])
    valset = LENADataSet(dataset_config["val"], **dataset_config["args"])
    sound, label = trainset[20]
    print(len(trainset), len(valset))
    print(sound.shape, label.shape)
    for i in range(100):
        sound, label = trainset[i]
        print(label[:, :30], '\n')
        print(label[:, 50:], '\n')
