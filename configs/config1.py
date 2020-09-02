from data import LENADataSet
device_ids = [0]
train_LENA = '/home/joseph/Desktop/LENA_praat/train.scp'
test_LENA = '/home/joseph/Desktop/LENA_praat/test.scp'
sr = 16000
datasets_train = [LENADataSet(train_LENA, ignore_duration=0.0, segment_len=32, stride=16, sr=16000, minmax=True)
                    ]
datasets_val = [LENADataSet(test_LENA, ignore_duration=0.0, segment_len=32, stride=16, sr=16000, minmax=True)
                    ]                