from data import LENADataSet
import os
import time
device_ids = [0]
root = '/ws/ifp-10_3/hasegawa/junzhez2'
config = os.path.basename(__file__).split('.')[0]
train_LENA = os.path.join(root, 'LENA/train.scp')
test_LENA = os.path.join(root, 'LENA/test.scp')
sr = 16000
num_spks = 2
use_mask = False
device_ids = [3]
spkr2idx = {'CHN':1, 'CXN':1, 'FAN':1, 'MAN':1}
# in_channels, out_channels, kernel_size, stride, bn, mp, dropout, relu,  res
layers = [(1,        80,         41,         3, True),
          (80,       160,        7,          1, True, 3),
          (160,      160,        3),
          (160,      160,        3,          3),
          (160,      160,        3,          1, True, 3),
          (160,      160,        3,          3),
          (160,      160,        3,          1, True, 3),
          (160,      160,        13),
          (160,      2048,       15),
          (2048,     2048,       1,          1, False, 1,   0.3),
          (2048,     2048,       1,          1, False, 1,   0.3),
          (2048,     num_spks,   1,          1, False, 1,   0.0,  False, False)]
datasets_train = [LENADataSet(train_LENA, segment_len=32, stride=16, sr=16000, minmax=True, layers=layers, device_ids=device_ids, use_mask=use_mask, spkr2idx=spkr2idx)
                    ]
datasets_val = [LENADataSet(test_LENA, segment_len=32, stride=16, sr=16000, minmax=True, layers=layers, device_ids=device_ids, use_mask=use_mask, spkr2idx=spkr2idx)
                    ]   
step_size = 1
gamma = 0.98
epochs = 128
save_folder = os.path.join(root, 'MaxMin_Pytorch', 'models')
checkpoint = 1
continue_from = os.path.join(save_folder, config + ".pth")
model_path = config + '_best.pth'
print_freq = 10
early_stop = True
max_norm = 5
lr = 1e-4
lr_override = False
comment = 'VAD'
log_dir = os.path.join(root, 'MaxMin_Pytorch', 'runs', time.strftime("%Y%m%d-%H%M%S") + config + comment)
shuffle = True