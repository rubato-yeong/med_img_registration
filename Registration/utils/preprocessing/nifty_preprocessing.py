'''
    nifty_preprocessing.py
        Main Function: nif2npy
'''

import os
import numpy as np
import nibabel as nib

from tqdm.notebook import tqdm
from scipy.ndimage import zoom



'''
    Support Functions
        voxel2resize
'''


def voxel2resize(voxel, shape, report=None):
    resize_shape = [shape[i]/voxel.shape[i] for i in range(3)]
    voxel = zoom(voxel, resize_shape)
    if report is not None:
        if voxel.shape != tuple(shape):
            print ('[{}] Error Shape:'.format(report), voxel.shape)
    return voxel



'''
    nif2npy

        1. Description

            Convert All NifTi Files to Numpy Voxel Files

        2. Folders

            CT_target : [0, 1] Normalized, [ShapeList] Resized CT Voxels
            PT_vanilla : [0, 1] Normalized, [ShapeList] Resized PT Voxels
            PT_resize : [0, 1] normalized, [ShapeList] Resized, [PixelSpacing] Corrected PT Voxels

            ->  Normalization:
                    CT : [-1024, 3071]      -> [0, 1]
                    PT : [0, 255] (Clipped) -> [0, 1]        

            ->  Compare:
                    [CT_target & PT_vanilla]
                    [CT_target & PT_resize]
                to check the effect of preprocessing.

        3. Inputs

            DATA_PATH : ../nifty   (NifTi FILES)
            SAVE_PATH : ../numpy   (numpy FILES)
        

'''


def nif2npy(PATH,
            ShapeList=[128,128,256]):

    DATA_PATH = PATH + '/nifty'
    SAVE_PATH = PATH + '/numpy'
    CT_PATH_LIST = sorted(os.listdir(os.path.join(DATA_PATH, 'CT')))
    PT_PATH_LIST = sorted(os.listdir(os.path.join(DATA_PATH, 'PT')))

    for i in tqdm(range(len(CT_PATH_LIST)), desc='Numpy Processing'):
        
        # NifTi Path
        CT_FILE_PATH = CT_PATH_LIST[i]
        PT_FILE_PATH = PT_PATH_LIST[i]
        CT_PATH = os.path.join(DATA_PATH + '/CT', CT_FILE_PATH)
        PT_PATH = os.path.join(DATA_PATH + '/PT', PT_FILE_PATH)
        PATIENT_NUM = int(CT_FILE_PATH[:2])

        # Numpy Path
        CT_SAVE_PATH = SAVE_PATH + '/CT_target'
        PT_SAVE_PATH = SAVE_PATH + '/PT_vanilla'
        PT_SAVE_PATH_2 = SAVE_PATH + '/PT_resize'
        if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)
        if not os.path.exists(CT_SAVE_PATH): os.mkdir(CT_SAVE_PATH)
        if not os.path.exists(PT_SAVE_PATH): os.mkdir(PT_SAVE_PATH)
        if not os.path.exists(PT_SAVE_PATH_2): os.mkdir(PT_SAVE_PATH_2)

        # NifTi to Numpy
        CT_nii, PT_nii = nib.load(CT_PATH), nib.load(PT_PATH)
        CT_voxel, PT_voxel = CT_nii.get_fdata(), PT_nii.get_fdata()

        # PET Pixel Correction
        TotalSpacing = [0, 0]
        CTPixelSpacing = CT_nii.header['pixdim'][1:3]
        PTPixelSpacing = PT_nii.header['pixdim'][1:3]
        TotalSpacing[0] = int(CT_voxel.shape[0] * (CTPixelSpacing[0]/PTPixelSpacing[0]))
        TotalSpacing[1] = int(CT_voxel.shape[1] * (CTPixelSpacing[1]/PTPixelSpacing[1]))
        Pad_X = int((PT_voxel.shape[0] - TotalSpacing[0])/2)
        Pad_Y = int((PT_voxel.shape[1] - TotalSpacing[1])/2)
        PT_resize = PT_voxel[Pad_X:Pad_X+TotalSpacing[0], Pad_Y:Pad_Y+TotalSpacing[1], :]

        # Resizing
        CT_voxel = voxel2resize(CT_voxel, ShapeList)
        PT_voxel = voxel2resize(PT_voxel, ShapeList)
        PT_resize = voxel2resize(PT_resize, ShapeList)

        # Normalization
        CT_voxel = np.clip(CT_voxel, -1024, 3071)
        PT_voxel = np.clip(PT_voxel, 0, 255)
        PT_resize = np.clip(PT_resize, 0, 255)
        CT_voxel = (CT_voxel + 1024) / 4095
        PT_voxel = PT_voxel / 255

        # Save
        np.save(os.path.join(CT_SAVE_PATH, '{:03d}'.format(PATIENT_NUM)), CT_voxel)
        np.save(os.path.join(PT_SAVE_PATH, '{:03d}'.format(PATIENT_NUM)), PT_voxel)
        np.save(os.path.join(PT_SAVE_PATH_2, '{:03d}'.format(PATIENT_NUM)), PT_resize)