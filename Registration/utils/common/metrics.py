'''
    metrics.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCC(nn.Module):

    def __init__(self, device):

        super().__init__()
        self.device = device
    
    def forward(self, y_pred, y_true):
        
        # assumes I, J are sized [batch_size, nb_feats=1, *vol_shape]
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims 

        # set window size
        win = Ii.shape[2:]

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # Sum
        stride, padding = 1, 0
        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_std = torch.sqrt(I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size)
        J_std = torch.sqrt(J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size)

        cc = cross / (I_std * J_std)

        # Squeeze to [N,]
        if cc.shape[0] == 1: cc = cc.squeeze().unsqueeze(0)
        else: cc = cc.squeeze()

        return cc