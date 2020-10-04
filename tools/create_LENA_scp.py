import glob
import random
import os
import json
LENA_folder = "/home/joseph/Desktop/LENA/OLD_PROTOCOL"
wavfiles = glob.glob(os.path.join(LENA_folder, '**/*.wav'), recursive = True)
wavfiles.sort()
random.seed(0)
random.shuffle(wavfiles)
trainfiles = wavfiles[:87]
valfiles = wavfiles[87:97]
testfiles = wavfiles[97:107]

def findmap(wavfile):
    try:
        annofile = glob.glob(wavfile.replace('.wav', '*.TextGrid'))[0]
        return annofile
    except:
        annofile = glob.glob(wavfile.replace('.wav', '*.textgrid'))[0]
        return annofile

with open(os.path.join(LENA_folder, 'train.scp'), 'w+') as file:
    dict = {}
    for trainfile in trainfiles:
        dict[trainfile] = findmap(trainfile)
    json.dump(dict, file)
with open(os.path.join(LENA_folder, 'val.scp'), 'w+') as file:
    dict = {}
    for valfile in valfiles:
        dict[valfile] = findmap(valfile)
    json.dump(dict, file)
with open(os.path.join(LENA_folder, 'test.scp'), 'w+') as file:
    dict = {}
    for testfile in testfiles:
        dict[testfile] = findmap(testfile)
    json.dump(dict, file)

print(len(trainfiles), len(valfiles), len(testfiles))
