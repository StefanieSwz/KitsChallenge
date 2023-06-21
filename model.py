# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:48:57 2023

@author: st_sc
"""

import sys
import os



import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
#data_path = '..\..\kits23\dataset'
#os.chdir(data_path)

#example_filename = 'case_00000/imaging.nii.gz'
import glob
import nibabel as nib
from nilearn import plotting
import numpy as np
import torch
from torchvision.transforms import ToTensor


# Define the path pattern to match the NIfTI files
path_pattern = '**/imaging.nii.gz'

# Use glob to find all file paths that match the pattern
file_paths = glob.glob(path_pattern, recursive=True)
nifti_image = []
# Iterate over the file paths and load each NIfTI file
for file_path in file_paths:
    nifti_image.append(nib.load(file_path))
    # Perform operations on the loaded image as needed
    # For example, you can access the image data using nifti_image.get_fdata()
    # or access the image dimensions using nifti_image.shape




# Get the image data as a NumPy array
image = nib.load("case_00000/imaging.nii.gz")
image_array = nifti_image.get_fdata()
slize = image_array[:, :, 50]
# Display the image using matplotlib
plt.imshow(image_array[100], cmap = 'bone')  # Display a single slice (e.g., slice 50), 
plt.axis('off')  # Turn off axes
plt.show()

#%% 

test_im = nifti_image[0]
test_im = test_im.get_fdata()
test_tensor = test_im.ToTensor()
