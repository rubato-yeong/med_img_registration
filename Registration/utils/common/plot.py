'''
    Plot.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

'''
    Functions
        mid_plot : Middle Part Visualization (unimodal)
        compare_plot : Compare Visualization (multimodal)
'''

def mid_plot(PATH,
             grid,
             fig=None,
             view=0, 
             cmap=plt.cm.gray, 
             vmax=None):

    if fig is None:
        fig = plt.figure(figsize=(15,15))
    axes = []
 
    for i, NPY_PATH in enumerate(os.listdir(PATH)):
        voxel = np.load(os.path.join(PATH, NPY_PATH))
        mid_len = int(voxel.shape[view]/2)
        axes.append(fig.add_subplot(grid[0], grid[1], i+1))
        axes[-1].set_title(NPY_PATH, fontsize=15)
        if view == 0:
            plt.imshow(voxel[mid_len,:,::-1].T, aspect='auto', cmap=cmap, vmax=vmax, vmin=0)
        elif view == 1:
            plt.imshow(voxel[:,mid_len,::-1].T, aspect='auto', cmap=cmap, vmax=vmax, vmin=0)
        else:
            plt.imshow(voxel[:,:,mid_len],      aspect='auto', cmap=cmap, vmax=vmax, vmin=0)


def compare_plot(PATH_1, PATH_2,
                 grid,
                 fig=None,
                 view=0,
                 cmap=[plt.cm.gray, plt.cm.hot],
                 vmax=[None, None]):
    
    if fig is None:
        fig = plt.figure(figsize=(15,15))
    axes = []

    PATH_1_list = os.listdir(PATH_1)
    PATH_2_list = os.listdir(PATH_2)
    plot_len = min(len(PATH_1_list), grid[0]*grid[1])
    
    for i in range(plot_len):

        voxel_1 = np.load(os.path.join(PATH_1, PATH_1_list[i]))
        voxel_2 = np.load(os.path.join(PATH_2, PATH_2_list[i]))
        mid_len_1 = int(voxel_1.shape[view]/2)
        mid_len_2 = int(voxel_2.shape[view]/2)

        axes.append(fig.add_subplot(grid[0], grid[1], 2*i+1))
        axes[-1].set_title(PATH_1_list[i], fontsize=15)
        if view == 0:   plt.imshow(voxel_1[mid_len_1,:,::-1].T, aspect='auto', cmap=cmap[0], vmax=vmax[0], vmin=0)
        elif view == 1: plt.imshow(voxel_1[:,mid_len_1,::-1].T, aspect='auto', cmap=cmap[0], vmax=vmax[0], vmin=0)
        else:           plt.imshow(voxel_1[:,:,mid_len_1],      aspect='auto', cmap=cmap[0], vmax=vmax[0], vmin=0)

        axes.append(fig.add_subplot(grid[0], grid[1], 2*i+2))
        axes[-1].set_title(PATH_2_list[i], fontsize=15)
        if view == 0:   plt.imshow(voxel_2[mid_len_2,:,::-1].T, aspect='auto', cmap=cmap[1], vmax=vmax[1], vmin=0)
        elif view == 1: plt.imshow(voxel_2[:,mid_len_2,::-1].T, aspect='auto', cmap=cmap[1], vmax=vmax[1], vmin=0)
        else:           plt.imshow(voxel_2[:,:,mid_len_2],      aspect='auto', cmap=cmap[1], vmax=vmax[1], vmin=0)
