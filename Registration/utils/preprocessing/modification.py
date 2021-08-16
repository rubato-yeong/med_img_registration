'''
    modification.py
'''

import os
import numpy as np
from tqdm.notebook import tqdm

def clip_voxel(PATH, limit, log=False):

    LOAD_PATH = PATH + '/numpy/PT_resize'
    if log: SAVE_PATH = PATH + '/numpy/PT_logclip'
    else: SAVE_PATH = PATH + '/numpy/PT_clip'
    if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)
    file_list = sorted(os.listdir(LOAD_PATH))
    minimum, maximum = limit[0], limit[1]

    for i in tqdm(range(len(file_list)), desc='Clipping'):
        fname = file_list[i]
        voxel = np.load(os.path.join(LOAD_PATH, fname))
        if log:
            voxel = (np.log(np.clip(255 * voxel, *limit) + 1e-1) - np.log(minimum + 1e-1)) / (np.log(maximum + 1e-1) - np.log(minimum + 1e-1))
        else:
            voxel = (np.clip(255 * voxel, *limit) - minimum) / (maximum - minimum)
        np.save(os.path.join(SAVE_PATH, fname), voxel)