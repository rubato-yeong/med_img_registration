import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model.losses as losses
import utils.common.metric as metric

# PATH Setting
DIR_PATH = 'F:/20210730_samples'
SAVE_PATH = os.path.join(DIR_PATH, 'numpy')
CT_PATH = os.path.join(SAVE_PATH, 'CT_target')
PT_PATH = os.path.join(SAVE_PATH, 'PT_clip')

device = torch.device('cuda')

Ii = np.zeros([16, 1, 128, 128, 256])
Ji = np.zeros([16, 1, 128, 128, 256])
for i in range(16):
    Ii[i,0] = np.load(os.path.join(CT_PATH, os.listdir(CT_PATH)[i]))
    Ji[i,0] = np.load(os.path.join(PT_PATH, os.listdir(PT_PATH)[i]))

Ii = torch.FloatTensor(Ii).to(device)
Ji = torch.FloatTensor(Ji).to(device)

start = time.time()

criterion = metric.NMI(num_bin=8, device=device).to(device)
cc = criterion(Ii[:2], Ji[:2]).cpu()
print ('Autograd Check', cc.requires_grad)
print ('NCC', cc)
print ('Average NCC', cc.mean().item())
print ('Time:', time.time()-start)
