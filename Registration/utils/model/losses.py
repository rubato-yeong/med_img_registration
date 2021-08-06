'''
    losses.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCC(nn.Module):
    '''
        Normalized Cross Correlation
    '''

    def __init__(self,
                 local=True, 
                 metric=False,
                 win=None,
                 eps=1e-5,
                 device=torch.device('cuda')):
        '''
            win : local window size (e.g. (16, 16, 16))
            win_default (9) : default local window size (9, 9, 9)
            local : True -> local / False -> global calculation
            metric : True -> metric / False -> loss calculation
            eps : epsilon constant (prevent division by 0)
        '''

        super().__init__()

        # Initial Values
        self.device = device
        self.local = local
        self.metric = metric
        self.win = None if win is None else np.array(win)
        self.win_size = None
        self.win_default = 9
        self.eps = eps

        # Internal Variables
        self.ndims = None
        self.conv_fn = None
        self.stride = 1
        self.padding = None

    
    def _loss(self, Ii, Ji):
        
        cross, I_var, J_var = self._ncc(Ii, Ji)
        cc = cross * cross / (I_var * J_var + self.eps) # [bs, 1, *V]

        return 1 - cc
        

    def _metric(self, Ii, Ji):

        cross, I_var, J_var = self._ncc(Ii, Ji)
        I_std, J_std = torch.sqrt(I_var), torch.sqrt(J_var)
        cc = cross / (I_std * J_std + self.eps) # [bs, 1, *V]

        return cc


    def _ncc(self, Ii, Ji):

        # compute filters
        sum_filt = torch.ones([1, 1, *self.win]).to(self.device)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % self.ndims)

        # compute NCC
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # sum
        I_sum = conv_fn(Ii, sum_filt, stride=self.stride, padding=self.padding)
        J_sum = conv_fn(Ji, sum_filt, stride=self.stride, padding=self.padding)
        I2_sum = conv_fn(I2, sum_filt, stride=self.stride, padding=self.padding)
        J2_sum = conv_fn(J2, sum_filt, stride=self.stride, padding=self.padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=self.stride, padding=self.padding)

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        return cross, I_var, J_var

    
    def forward(self, y_pred, y_true):
        
        # assumes I, J are sized [batch_size, nb_feats=1, *vol_shape]
        Ii, Ji = y_true, y_pred

        # get dimension of volume
        self.ndims = len(Ii.shape) - 2
        error_msg = "volumes should be 1 to 3 dimensions. found: %d" % self.ndims
        assert self.ndims in [1, 2, 3], error_msg

        # set window size
        if self.local:
            if self.win is None: self.win = np.full(self.ndims, self.win_default)
            self.padding = tuple(np.floor(self.win/2).astype('int32'))
            self.win_size = np.prod(self.win)
        else:
            self.win = Ii.shape[2:]
            self.padding = 0
            self.win_size = np.prod(self.win)
        
        # get ncc tensor
        if self.metric: cc = self._metric(Ii, Ji)
        else: cc = self._loss(Ii, Ji)
        
        # [bs, 1, *V] -> [bs]
        cc = cc.reshape(cc.shape[0], -1)

        return torch.mean(cc, dim=1)
        

        


