'''
    dicom_preprocessing.py
        Main Function: path2base
'''

import os
import numpy as np
import pydicom

from tqdm.notebook import tqdm
from scipy.ndimage import zoom




'''
    Support Functions
        path2dcm
        dcm2voxel
        voxel2resize
'''

def path2dcm(PATH):
    DCM_REF = ['IMG-0001-00001.dcm', 'IMG-0002-00001.dcm', 'IMG-0003-00001.dcm']
    IDX_REF = [None, None]
    CT_FILES, PT_FILES = [], []

    for REF in DCM_REF:
        REF_PATH = os.path.join(PATH, REF)
        dicom_ref = pydicom.dcmread(REF_PATH)
        if dicom_ref.SeriesDescription == 'PRE_CT_TORSO':
            IDX_REF[0] = REF[:8]
        elif dicom_ref.SeriesDescription == 'PET ax':
            IDX_REF[1] = REF[:8]
    
    if IDX_REF[0] is None: print('CT DATA EMPTY')
    if IDX_REF[1] is None: print('PT DATA EMPTY')

    for DCM in os.listdir(PATH):
        DCM_PATH = os.path.join(PATH, DCM)
        if DCM[:8] == IDX_REF[0]: #CT
            CT_FILES.append(pydicom.dcmread(DCM_PATH))
        elif DCM[:8] == IDX_REF[1]: #PT
            PT_FILES.append(pydicom.dcmread(DCM_PATH))
    return CT_FILES, PT_FILES[::-1]


def dcm2voxel(FILES):
    dsRef = FILES[0]
    dims = (int(dsRef.Rows), int(dsRef.Columns), len(FILES))
    voxel = np.zeros(dims)
    for i, DCM in enumerate(FILES):
        voxel[:,:,i] = DCM.pixel_array
    return voxel


def voxel2resize(voxel, shape, report=None):
    resize_shape = [shape[i]/voxel.shape[i] for i in range(3)]
    voxel = zoom(voxel, resize_shape)
    if report is not None:
        if voxel.shape != tuple(shape):
            print ('[{}] Error Shape:'.format(report), voxel.shape)
    return voxel




'''
    path2base

        1. Description

            Convert All DICOM Files to Numpy Voxel Files

        2. Folders

            CT_target : [0, 1] Normalized, [ShapeList] Resized CT Voxels
            PT_vanilla : [0, 1] Normalized, [ShapeList] Resized PT Voxels
            PT_resize : [0, 1] normalized, [ShapeList] Resized, [PixelSpacing] Corrected PT Voxels

            ->  Compare:
                    [CT_target & PT_vanilla]
                    [CT_target & PT_resize]
                to check the effect of preprocessing.

        3. Inputs

            DATA_PATH : ../rawdata (DICOM FILES)
            SAVE_PATH : ../numpy   (NUMPY FILES)
        

'''

def path2base(DATA_PATH, SAVE_PATH,
              ShapeList=[128, 128, 256],
              RangeList=[4095, 32767]):

    for i in tqdm(range(len(os.listdir(DATA_PATH))), desc='Numpy Processing'):
        PATIENT_PATH = os.listdir(DATA_PATH)[i]

        if len(PATIENT_PATH) <= 3:

            # DICOM Path Check
            patient_num = int(PATIENT_PATH[:int(PATIENT_PATH.find('a'))])
            PATIENT_PATH = os.path.join(DATA_PATH, PATIENT_PATH)
            PATIENT_PATH = os.path.join(PATIENT_PATH, os.listdir(PATIENT_PATH)[0])

            # Numpy Path Check
            CT_PATH = SAVE_PATH + '/CT_target'
            PT_OLD_PATH = SAVE_PATH + '/PT_vanilla'
            PT_NEW_PATH = SAVE_PATH + '/PT_resize'
            if not os.path.exists(CT_PATH): os.mkdir(CT_PATH)
            if not os.path.exists(PT_OLD_PATH): os.mkdir(PT_OLD_PATH)
            if not os.path.exists(PT_NEW_PATH): os.mkdir(PT_NEW_PATH)

            # Path to Numpy Voxels
            ct_files, pt_files = path2dcm(PATIENT_PATH)
            ct_voxel, pt_voxel = dcm2voxel(ct_files), dcm2voxel(pt_files)

            # PET Pixel Correction
            TotalSpacing = [0, 0]
            CTPixelSpacing = ct_files[0].PixelSpacing
            PTPixelSpacing = pt_files[0].PixelSpacing
            TotalSpacing[0] = int(ct_voxel.shape[0] * (CTPixelSpacing[0]/PTPixelSpacing[0]))
            TotalSpacing[1] = int(ct_voxel.shape[1] * (CTPixelSpacing[1]/PTPixelSpacing[1]))
            Pad_X = int((pt_voxel.shape[0] - TotalSpacing[0])/2)
            Pad_Y = int((pt_voxel.shape[1] - TotalSpacing[1])/2)
            pt_resize = pt_voxel[Pad_X:Pad_X+TotalSpacing[0], Pad_Y:Pad_Y+TotalSpacing[1], :]

            # Resizing
            ct_voxel = voxel2resize(ct_voxel, ShapeList)
            pt_voxel = voxel2resize(pt_voxel, ShapeList)
            pt_resize = voxel2resize(pt_resize, ShapeList)

            # Clip
            ct_voxel = np.clip(ct_voxel, 0, RangeList[0])
            pt_voxel = np.clip(pt_voxel, 0, RangeList[1])
            pt_resize = np.clip(pt_resize, 0, RangeList[1])

            # Normalization
            ct_voxel = ct_voxel / RangeList[0]
            pt_voxel = pt_voxel / RangeList[1]
            pt_resize = pt_resize / RangeList[1]

            # Save
            np.save(os.path.join(CT_PATH, '{:03d}'.format(patient_num)), ct_voxel)
            np.save(os.path.join(PT_OLD_PATH, '{:03d}'.format(patient_num)), pt_voxel)
            np.save(os.path.join(PT_NEW_PATH, '{:03d}'.format(patient_num)), pt_resize)




if __name__ == "__main__":
    print("This is standardization.py")
    print("This Code should be running on Jupyter Notebook.")