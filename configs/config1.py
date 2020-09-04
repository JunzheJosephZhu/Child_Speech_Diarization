from data import LENADataSet
device_ids = [0]
train_LENA = '/home/joseph/Desktop/LENA_praat/train.scp'
test_LENA = '/home/joseph/Desktop/LENA_praat/test.scp'
sr = 16000
datasets_train = [LENADataSet(train_LENA, ignore_duration=0.0, segment_len=32, stride=16, sr=16000, minmax=True)
                    ]
datasets_val = [LENADataSet(test_LENA, ignore_duration=0.0, segment_len=32, stride=16, sr=16000, minmax=True)
                    ]   
num_spks = 3             
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
lr = 0.001
step_size = 1
gamma = 0.98