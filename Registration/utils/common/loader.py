import numpy as np
import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

class Data_Generator(Dataset):
    def __init__(self, moving_root, fixed_root):
        # File Directory
        self.moving_files = sorted(list(Path(moving_root).iterdir()))
        self.fixed_files = sorted(list(Path(fixed_root).iterdir()))
        # Extract Features
        example = np.load(self.moving_files[0])
        self.vol_shape = example.shape
        self.ndims = len(self.vol_shape)        
    
    def __len__(self):
        return len(self.moving_files)
    
    def __getitem__(self, idx):
        # Numpy Load
        fname = self.moving_files[idx].name
        moving_img = np.load(self.moving_files[idx])
        fixed_img = np.load(self.fixed_files[idx])
        moved_img = fixed_img.copy()
        # Tensor Conversion
        moving_img = torch.FloatTensor(moving_img).unsqueeze(0)
        fixed_img = torch.FloatTensor(fixed_img).unsqueeze(0)
        moved_img = torch.FloatTensor(moved_img).unsqueeze(0)
        zero_phi = torch.zeros([self.ndims, *self.vol_shape])
        # Output
        return moving_img, fixed_img, moved_img, zero_phi, fname



