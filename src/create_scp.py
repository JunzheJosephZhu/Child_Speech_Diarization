import glob
import random
import os
import json
LENA_folder = "/ws/ifp-10_3/hasegawa/junzhez2/LENA"
wavfiles = glob.glob(os.path.join(LENA_folder, '**/*.wav'), recursive = True)
wavfiles.sort()
random.seed(0)
random.shuffle(wavfiles)
trainfiles = wavfiles[:80]
testfiles = wavfiles[80:]
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
with open(os.path.join(LENA_folder, 'test.scp'), 'w+') as file:
    dict = {}
    for testfile in testfiles:
        dict[testfile] = findmap(testfile)
    json.dump(dict, file)

print(len(trainfiles), len(testfiles))
