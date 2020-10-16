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
lena_label_root = Path("/home/joseph/Desktop/LENA_sys")
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


with open(human_label_root / "test.scp") as file:
    data = json5.load(file)
error_total, base_total = defaultdict(float), defaultdict(float)
for wavfile, annofile in data.items():
    # human label
    label_human = get_label(wavfile, annofile)
    label_human = torch.Tensor(label_human).unsqueeze(0)

    # machine label
    annofile_tmp = annofile.replace(str(human_label_root), str(lena_label_root))
    annofile_tmp = "_".join(annofile_tmp.split("_")[:-1])
    lenafile = annofile_tmp + ".textgrid"
    assert os.path.exists(lenafile), (annofile, lenafile)
    label_lena = get_label(wavfile, lenafile)
    label_lena = torch.Tensor(label_lena).unsqueeze(0)

    for name in metrics:
        error, base = getattr(metric, name)(median_filter=1)(label_lena - 0.5, label_human)
        error_total[name] += error
        base_total[name] += base

print({name: error_total[name] / base_total[name] for name in metrics})
