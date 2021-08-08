'''

    losses.py

    Reference
    [1] https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/losses.py
    [2] https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/main/ViT-V-Net/losses.py

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
    Normalized Cross Correlation
'''

class NCC(nn.Module):

    def __init__(self,
                 local=True, 
                 metric=False,
                 win=None,
                 eps=1e-5,
                 device='cuda'):
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
        self.first = True
        self.ndims = None
        self.conv_fn = None
        self.filter = None
        self.stride = 1
        self.padding = None


    def _setup(self, vol_shape):
        
        # get dimension of volume
        self.ndims = len(vol_shape) - 2
        
        # get convolution function
        self.conv_fn = getattr(F, 'conv%dd' % self.ndims)

        # set window size
        if self.local:
            if self.win is None: self.win = np.full(self.ndims, self.win_default)
            self.padding = tuple(np.floor(self.win/2).astype('int32'))
            self.win_size = np.prod(self.win)
        else:
            self.win = vol_shape[2:]
            self.padding = 0
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

        if self.metric:
            I_std, J_std = torch.sqrt(I_var), torch.sqrt(J_var)
            cc = cross / (I_std * J_std + self.eps) # [bs, 1, *V]
        else:
            cc = cross * cross / (I_var * J_var + self.eps) # [bs, 1, *V]
            cc = 1 - cc

        return cc

    
    def forward(self, y_pred, y_true):

        # setup if first
        if self.first:
            self._setup(vol_shape=y_true.shape)
            self.first = False
        
        # get ncc tensor, sized [bs, 1, *V]
        cc = self._ncc(y_true, y_pred)
        cc = cc.reshape(cc.shape[0], -1)

        return torch.mean(cc, dim=1)




'''
    Normalized Mutual Information (on Voxels)
'''

class NMI(torch.nn.Module):

    def __init__(self,
                 metric=False,
                 num_bin=16,
                 vmin=0.,
                 vmax=1.,
                 sigma_ratio=0.1,
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

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.metric = metric
        self.eps = eps
        self.device = device

    def _nmi(self, y_true, y_pred):

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        # compute image terms by approx. Gaussian dist
        I_a = torch.exp(- self.preterm * torch.square(y_true - self.vol_bin_centers))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True) # [bs, V, B]

        I_b = torch.exp(- self.preterm * torch.square(y_pred - self.vol_bin_centers))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True) # [bs, V, B]

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b) / nb_voxels # [bs, B, B]
        pa = torch.mean(I_a, dim=1, keepdim=False) # [bs, B]
        pb = torch.mean(I_b, dim=1, keepdim=False) # [bs, B]

        Hxy = - torch.sum(torch.sum(pab * torch.log2(pab + self.eps), axis=1), axis=1)
        Hx = - torch.sum(pa * torch.log2(pa + self.eps), axis=1)
        Hy = - torch.sum(pb * torch.log2(pb + self.eps), axis=1)

        nmi = 2 * (1 - Hxy / (Hx + Hy)) # [bs]

        if self.metric: return nmi
        else: return 1 - nmi 

    def forward(self, y_true, y_pred):

        return self._nmi(y_true, y_pred)
    

'''

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=16, patch_size=5):
        super().__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, steps=num_bin)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)
        
        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()
        
        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)
        
        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size,
                                            (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size,
                                            (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))
        
        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)
        
        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self,y_true, y_pred):
        return -self.local_mi(y_true, y_pred)


'''