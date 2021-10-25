#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:20:30 2021

@author: dma73
"""

cam_map_array = cam_map_arrays[e]
for i in range(cam_map_array.shape[-1]):
  cam_map = cam_map_array[...,i]
  print(cam_map.min(),cam_map.max())

  cam_map[cam_map<-0.1] = -0.1
  cam_map[cam_map>0.1] = 0.1
  
  
  plt.subplot(3,1,i+1)
  plt.imshow(cam_map_array[...,i].max(axis=0).transpose([1,0]), extent=(0,*imsize,0), alpha=0.6, cmap='magma', interpolation='bilinear')
  plt.hist(cam_map_array[...,i].flatten())
  plt.xlim([-0.1,0.1])

#%%
thres = 0.2
cam_map_array_thre = np.copy(cam_map_array)
cam_map_array_thre[cam_map_array_thre<-thres] = -thres
cam_map_array_thre[cam_map_array_thre>thres] = thres

# f = v.slice_viewer(cam_map_array_thre.max(axis=0))

multi_slice_viewer(np.transpose(cam_map_array_thre.max(axis=0),[2,1,0]))

plt.imshow(cam_map_array_thre.max(axis=0).std(axis=-1).transpose([1,0]), extent=(0,*imsize,0), alpha=0.6, cmap='jet', interpolation='bilinear')

plt.imshow(cam_map_array.max(axis=0).std(axis=-1).transpose([1,0]), extent=(0,*imsize,0), alpha=0.6, cmap='jet', interpolation='bilinear')

plt.imshow(uncertainty_map, extent=(0,*imsize,0), alpha=0.6, cmap='jet', interpolation='bilinear')

#%%
from importlib import reload
reload(v)

#%%
X = np.random.rand(20, 20, 40)
v.slice_viewer(X)

#%%
astronaut = data.astronaut()
ihc = data.immunohistochemistry()
hubble = data.hubble_deep_field()

# Initialize the subplot panels side by side
fig, ax = plt.subplots(nrows=1, ncols=3)

# Show an image in each subplot
ax[0].imshow(astronaut)
ax[0].set_title('Natural image')
ax[1].imshow(ihc)
ax[1].set_title('Microscopy image')
ax[2].imshow(hubble)
ax[2].set_title('Telescope image');

#%%

import tempfile

# Create a temporary directory
d = tempfile.mkdtemp()

import os

# Return the tail of the path
os.path.basename('http://google.com/attention.zip')

from urllib.request import urlretrieve

# Define URL
url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip'

# Retrieve the data
fn, info = urlretrieve(url, os.path.join(d, 'attention.zip'))

import zipfile

# Extract the contents into the temporary directory we created earlier
zipfile.ZipFile(fn).extractall(path=d)

# List first 10 files
[f.filename for f in zipfile.ZipFile(fn).filelist[:10]]

#%% read hdr image
import nibabel

# Read the image 
struct = nibabel.load(os.path.join(d, 'attention/structural/nsM00587_0002.hdr'))

# Get a plain NumPy array, without all the metadata
struct_arr = struct.get_fdata()

#%%
plt.imshow(struct_arr[75], aspect=0.5)
#%%
struct_arr2 = struct_arr.T
plt.imshow(struct_arr2[34])

fig, ax = plt.subplots()
ax.imshow(struct_arr[..., 43])
#%%
from med_deeplearning.HelperFunctions.visualization.slice_scroll import multi_slice_viewer
multi_slice_viewer(cam_map_arrays[e])
