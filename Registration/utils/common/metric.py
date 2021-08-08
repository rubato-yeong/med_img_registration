'''

    metric.py

    Reference
    [1] https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/losses.py
    [2] https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/main/ViT-V-Net/losses.py

'''

import numpy as np
from numpy.lib.nanfunctions import nanmin
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
    Normalized Cross Correlation
'''

class NCC(nn.Module):

    def __init__(self,
                 eps=1e-5,
                 device='cuda'):
        '''
            eps : epsilon constant (prevent division by 0)
        '''

        super().__init__()

        # Initial Values
        self.device = device
        self.win = None
        self.win_size = None
        self.eps = eps

        # Internal Variables
        self.first = True
        self.ndims = None
        self.conv_fn = None
        self.filter = None
        self.stride = 1
        self.padding = 0


    def _setup(self, vol_shape):
        
        # get dimension of volume
        self.ndims = len(vol_shape) - 2
        
        # get convolution function
        self.conv_fn = getattr(F, 'conv%dd' % self.ndims)

        # set window size
        self.win = vol_shape[2:]
        self.win_size = np.prod(self.win)
        
        # compute filters
        self.filter = torch.ones([1, 1, *self.win]).to(self.device)


    def _ncc(self, Ii, Ji):

        # assumes I, J are sized [batch_size, nb_feats=1, *vol_shape]

        # compute NCC
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # sum
        I_sum = self.conv_fn(Ii, self.filter, stride=self.stride, padding=self.padding)
        J_sum = self.conv_fn(Ji, self.filter, stride=self.stride, padding=self.padding)
        I2_sum = self.conv_fn(I2, self.filter, stride=self.stride, padding=self.padding)
        J2_sum = self.conv_fn(J2, self.filter, stride=self.stride, padding=self.padding)
        IJ_sum = self.conv_fn(IJ, self.filter, stride=self.stride, padding=self.padding)

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        I_std, J_std = torch.sqrt(I_var), torch.sqrt(J_var)
        cc = cross / (I_std * J_std + self.eps) # [bs, 1, 1, 1, 1]

        return cc
    
    def forward(self, y_pred, y_true):

        # setup if first
        if self.first:
            self._setup(vol_shape=y_true.shape)
            self.first = False
        
        # get ncc tensor
        cc = self._ncc(y_true, y_pred)
        cc = cc.reshape(cc.shape[0], -1)

        return torch.mean(cc, dim=1)




'''
    Normalized Mutual Information (on Voxels)
'''

class NMI(torch.nn.Module):

    def __init__(self,
                 num_bin=16,
                 vmin=0.,
                 vmax=1.,
                 sigma_ratio=1,
                 eps=1e-6,
                 device='cuda'):

        super().__init__() 

        # Create bin centers
        bin_centers = np.linspace(vmin, vmax, num=num_bin)
        vol_bin_centers = torch.linspace(vmin, vmax, steps=num_bin, device=device)
        vol_bin_centers = vol_bin_centers.reshape(1, 1, -1)
        num_bins = len(bin_centers)

        # Sigma for Gaussian approx (RBF)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.bin_centers = bin_centers
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.eps = eps
        self.device = device

    def _nmi(self, y_true, y_pred):

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        # compute image terms by approx. Gaussian dist
        I_a = torch.square(y_true - self.vol_bin_centers) + self.eps # [bs, V, B]
        I_a = I_a / torch.min(I_a, dim=2, keepdim=True)[0]
        I_a[I_a != 1] = 0

        I_b = torch.square(y_pred - self.vol_bin_centers) + self.eps # [bs, V, B]
        I_b = I_b / torch.min(I_b, dim=2, keepdim=True)[0]
        I_b[I_b != 1] = 0

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b) / nb_voxels # [bs, B, B]
        pa = torch.mean(I_a, dim=1, keepdim=False) # [bs, B]
        pb = torch.mean(I_b, dim=1, keepdim=False) # [bs, B]

        Hxy = - torch.sum(torch.sum(pab * torch.log2(pab + self.eps), axis=1), axis=1)
        Hx = - torch.sum(pa * torch.log2(pa + self.eps), axis=1)
        Hy = - torch.sum(pb * torch.log2(pb + self.eps), axis=1)

        nmi = 2 * (1 - Hxy / (Hx + Hy)) # [bs]

        return nmi

    def forward(self, y_true, y_pred):

        return self._nmi(y_true, y_pred)