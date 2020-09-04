import sys
import torch
from stage1 import TemporalConv
sys.path.append("/home/joseph/Desktop/MaxMin_Pytorch")
from configs.config1 import *
import torch
tr_dataset = torch.utils.data.ConcatDataset(datasets_train)
cv_dataset = torch.utils.data.ConcatDataset(datasets_val)
tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle = False)
cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=1, shuffle = False)
model = TemporalConv(layers)
model = torch.nn.DataParallel(model, device_ids=device_ids)
optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.001,
                                )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
