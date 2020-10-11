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

def _pad(data, maxlen, stride):
    '''
        data: [num_samples]
        maxlen: in units of stride
    '''
    n_frames = _count_frames(len(data), stride)
    padded_data = np.zeros(maxlen * stride)
    padded_data[:len(data)] = data
    key_padding_mask = np.ones(maxlen, dtype=bool)
    key_padding_mask[:n_frames] = 0
    return padded_data, key_padding_mask


class LENADataSet(torch.utils.data.Dataset):
    def __init__(self, scp_file, dataset_root, chunk_size, stride, sr, minmax, target_channels, spkr2idx, duration_thres=0.1):
        '''
            data_dir: scp file
            chunk_size: in frames
            stride: in samples
            duration_thres: discard all chunks with percentage of speech duration below this
            both modes of encoder should have padding
        '''
        super().__init__()
        self.chunk_size = chunk_size
        with open(os.path.join(dataset_root, scp_file)) as file:
            self.data = json.load(file)
        sounds = []
        labels = []
        for wavfile in self.data:
            _sr, sound = read(wavfile)
            assert sr == _sr
            sound = np.float32(sound)
            num_frames = _count_frames(len(sound), stride)
            label = np.zeros((target_channels, num_frames), dtype=float)
            
            if minmax: # mimax norm
                _min, _max = np.amin(sound), np.amax(sound)
                sound = (sound - _min) / (_max - _min)
                sound = sound * 2 - 1 # scale to [-1, 1]
            else: # z normchunk_size, stride, sr, minmax, 
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
        assert len(self.sounds) == len(self.labels)

    def __len__(self):
        return len(self.sounds)

    def __getitem__(self, idx):
        return self.sounds[idx], np.zeros(self.chunk_size, dtype=bool), self.labels[idx]

class MILDataSet(torch.utils.data.Dataset):
    def __init__(self, scp_file, dataset_root, stride, sr, target_channels, spkr2idx, maxlen):
        '''
            data_dir: scp file
            chunk_size: in frames
            stride: in samples
            duration_thres: minlen, maxlen: in frames
            both modes of encoder should have padding
        '''
        super().__init__()
        self.maxlen = maxlen
        self.stride = stride
        self.sr = sr
        self.spkr2idx = spkr2idx
        self.target_channels = target_channels
        with open(os.path.join(dataset_root, scp_file)) as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wavfile, n_frames, tag = self.data[idx]
        label = np.zeros(self.target_channels, dtype=np.int64)
        label[self.spkr2idx[tag]] = 1
        _sr, sound = read(wavfile)
        assert self.sr == _sr
        sound = np.float32(sound)
        padded_sound, mask = _pad(sound, self.maxlen, self.stride)
        assert self.maxlen - mask.sum() == n_frames
        return padded_sound, mask, label

if __name__ == '__main__':
    test_LENA = True
    test_MIL = True
    # test LENADataset
    if test_LENA:
        root = "~/Desktop/MaxMin_Pytorch"
        config_filename = "configs/AE_RNN.json"
        with open(os.path.expanduser(os.path.join(root, config_filename))) as file:
            config = json.load(file)
        dataset_config = config["dataset"]
        trainset = LENADataSet(dataset_config["train"], **dataset_config["args"])
        valset = LENADataSet(dataset_config["val"], **dataset_config["args"])
        sound, mask, label = trainset[20]
        print(len(trainset), len(valset))
        print(sound.shape, mask.shape, label.shape)
        for i in range(100):
            sound, mask, label = trainset[i]
            assert mask.shape[0] == label.shape[1]

    # test MILDataset
    if test_MIL:
        root = "~/Desktop/MaxMin_Pytorch"
        config_filename = "configs/MIL_AE_RNN.json"
        with open(os.path.expanduser(os.path.join(root, config_filename))) as file:
            config = json.load(file)
        dataset_config = config["dataset"]
        trainset = MILDataSet(dataset_config["train"], **dataset_config["args"])
        valset = MILDataSet(dataset_config["val"], **dataset_config["args"])
        sound, mask, label = trainset[20]
        print(len(trainset), len(valset))
        print(sound.shape, mask.shape, label.shape)
        for i in range(100):
            sound, mask, label = trainset[i]
