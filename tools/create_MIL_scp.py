import glob
import random
import os
import json
from tqdm import tqdm
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

stride = 4096
min_frames = 5
max_frames = 40

clips_folder = ['/home/joseph/Desktop/Child Adult Classifier/Providence_clips',
                '/home/joseph/Desktop/Child Adult Classifier/Braunwald_clips']
classes = ['CHI', 'MOT', 'FAT']
files = []
for cls in classes:
    for folder in clips_folder:
        wavfiles = glob.glob(os.path.join(folder, cls, "*.wav"))
        for wavfile in wavfiles:
            _sr, sound = read(wavfile)
            assert _sr == 16000
            n_frames = len(sound) // stride + 1
            if len(sound) % 4096 == 0:
                n_frames -= 1
            if min_frames <= n_frames and n_frames <= max_frames:
                files.append((wavfile, n_frames, cls))
files = sorted(files, key=lambda x:x[0])
random.seed(0)
random.shuffle(files)
trainfiles = files[:int(0.8 * len(files))]
valfiles = files[int(0.8 * len(files)):]


with open(os.path.join('outputs', 'MIL_train.scp'), 'w+') as file:
    json.dump(trainfiles, file)
with open(os.path.join('outputs', 'MIL_val.scp'), 'w+') as file:
    json.dump(valfiles, file)

print(len(trainfiles), len(valfiles))
