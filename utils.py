# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:48:42 2023

@author: st_sc
"""
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import glob
import nibabel as nib
#from matplotlib import pylab as plt
# import nibabel as nib
# from nibabel import nifti1
# from nibabel.viewers import OrthoSlicer3D
# import matplotlib.pyplot as plt
import os
  
class KidneyData(Dataset):
    def __init__(self, data_dir = "", batch_size = 32, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        
        current_dir = os.getcwd()
        os.chdir(current_dir + data_dir) # setting root directory to read data
        
        # Create a full list for training image cases
        # Define the path pattern to match the NIfTI files
        path_pattern_im = '**/imaging.nii.gz'
        # Use glob to find all file paths that match the pattern
        file_paths_im = glob.glob(path_pattern_im, recursive=True)
        nifti_image = []
        
        # Iterate over the file paths and load each NIfTI file
        for file_path in file_paths_im:
            nifti_image.append(nib.load(file_path))
        
        # Create a full list for training segmentation cases
        path_pattern_seg = '**/segmentation.nii.gz'
        file_paths_seg = glob.glob(path_pattern_seg, recursive=True)
        nifti_segment = []
        
        # Iterate over the file paths and load each NIfTI file
        for file_path in file_paths_seg:
            nifti_segment.append(nib.load(file_path))
        
        self.image = nifti_image
        self.mask = nifti_segment
        
        os.chdir(current_dir)
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image = self.image[idx]
        mask = self.mask[idx]
        image = image.get_fdata()
        mask = mask.get_fdata()
        image = np.transpose(image, (1,2,0))
        mask = np.transpose(mask, (1,2,0))
        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask
 


trans = transforms.Compose([ToTensor()])
kidney = KidneyData(transforms=trans)

batch_size = 1
shuffle = True

# Create a DataLoader
dataloader = DataLoader(kidney, batch_size=batch_size, shuffle=shuffle)

batch = next(iter(dataloader))
