import sys
import torch
torch.manual_seed(0)
from classifier import TemporalConv
from solver import Solver
import os

root = '/ws/ifp-10_3/hasegawa/junzhez2/MaxMin_Pytorch'
sys.path.append(root)
sys.path.append(os.path.join(root, "configs"))

import argparse
parser = argparse.ArgumentParser(description='config file')
parser.add_argument('--config', type=str, default='config1', help='config file')
args = parser.parse_args()
exec('from ' + args.config + ' import *')
print('loading ' + args.config)

tr_dataset = torch.utils.data.ConcatDataset(datasets_train)
cv_dataset = torch.utils.data.ConcatDataset(datasets_val)
tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=shuffle)
cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=1, shuffle=shuffle)
data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
model = TemporalConv(layers)
model = torch.nn.DataParallel(model.cuda(device_ids[0]), device_ids=device_ids)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
solver = Solver(data, model, optimizer, scheduler, epochs, save_folder, checkpoint, continue_from, model_path, print_freq, 
                early_stop, max_norm, lr, lr_override, log_dir, config)
solver.train()

