import sys
sys.path.append('/home/joseph/Desktop/MaxMin_Pytorch/src')
from collections import defaultdict
import json5
import metric
from data import _count_frames
from praatio import tgio
from scipy.io.wavfile import read
import numpy as np
import os
from pathlib import Path
import torch
spkr2idx = {"CHN":0, "CXN":0, "FAN":1, "MAN":2}
sr = 16000
target_channels = 3
stride = 4096
human_label_root = Path("/home/joseph/Desktop/LENA/OLD_PROTOCOL")
marvin_label_root = Path("/home/joseph/Desktop/voice_type_classifier/output_voice_type_classifier")
ages = ["03m_IDP", "06m_IDP", "09m_IDP", "12m_IDP", "12m-24m_TDP"]
voice_types = {'CHI':0, "KCHI":0, "FEM":1, "MAL":2}
metrics =["DER", "Frame_ERR", "Frame_Tier_ERR", "DER_Tier", "ERR_Tier"]

def get_label(wavfile, annofile):
    _sr, sound = read(wavfile)
    num_frames = _count_frames(len(sound), stride)
    label = np.zeros((target_channels, num_frames), dtype=np.float32)
    tg = tgio.openTextgrid(annofile)
    for spkr, idx in spkr2idx.items():
        try:
            entries = tg.tierDict[spkr].entryList
        except:
            # print('no entry', spkr)
            continue
        for entry in entries: # create label
            st = int(entry.start * sr / stride)
            ed = int(entry.end * sr / stride)
            if ed > num_frames:
                ed = num_frames
            label[idx, st:ed] = 1 # use slicing for faster writings
    return label

def get_label_marv(wavfile):
    _sr, sound = read(wavfile)
    num_frames = _count_frames(len(sound), stride)
    label = np.zeros((target_channels, num_frames), dtype=np.float32)
    name, _ = os.path.splitext(os.path.basename(wavfile))
    age = wavfile.split('/')[-2]
    assert age in ages
    for voice_type, idx in voice_types.items():
        annofile = marvin_label_root / age / (voice_type + '.rttm')
        assert os.path.exists(annofile), annofile
        with open(marvin_label_root / annofile) as handle:
            linelist = handle.readlines()
        count = 0
        for line in linelist:
            _, name_tmp, _, st, dur, _, _, _, _, _ = line.split()
            st, ed = float(st), float(st) + float(dur)
            st = int(st * sr / stride)
            ed = int(ed * sr / stride)
            if name_tmp == name:
                count += 1
                label[idx, st:ed] = 1

    return label

with open(human_label_root / "test.scp") as file:
    data = json5.load(file)
error_total, base_total = defaultdict(float), defaultdict(float)
for wavfile, annofile in data.items():
    # human label
    label_human = get_label(wavfile, annofile)
    label_human = torch.Tensor(label_human).unsqueeze(0)

    # machine label
    label_marv = get_label_marv(wavfile)
    label_marv = torch.Tensor(label_marv).unsqueeze(0)

    for name in metrics:
        error, base = getattr(metric, name)(median_filter=1)(label_marv - 0.5, label_human)
        error_total[name] += error
        base_total[name] += base

print({name: error_total[name] / base_total[name] for name in metrics})

