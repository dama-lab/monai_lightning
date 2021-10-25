# -*- coding: utf-8 -*-
# Zscape configuration: /project/rrg-mfbeg-ad/faisal_work/data-processing/abacs/managed-by-vincentc/zscape_1
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
import os

#%% calculate structure volume for a single label
def cal_label_vol(array, id,  voxel_size):
  ''' calculate structural volume for a single label
  array: input 3d volume
  id:   id of structural label
  return: vol
  '''
  vox_no = array[array==id].size
  # calculate volume from voxel number
  vol = vox_no
  for dim in range(len(voxel_size)):
    vol = vol * voxel_size[dim]
  
  return vol

#%% calculate structural volume for an array of labels
def cal_label_vols_array(array, ids, voxel_size, vol_array=np.empty(0), verbose=False):
  ''' calculate structural volume for an array of labels
  array: input 3d volume
  id:    ids of structural labels
  return: vol array
  '''
  for id in ids:
    if verbose==True: print(f"label: {id}")
    vol = cal_label_vol(array, id,  voxel_size)
    # adding volume size to vol_array 
    vol_array = np.append(vol_array, vol)
    if verbose>1: print(vol_array)
  return vol_array
#%% 
def get_verte_slice_no(tissue_verte, verte_id):
  ''' extraxt a slab of volume based on verte_id
  tissue_verte: array contains tissue+vertebrate segmentations
  '''
  slice_ids = np.where(tissue_verte==verte_id)[2]
  return slice_ids

#%% 
def get_verte_slice_minmax(tissue_verte, verte_id):
  ''' extraxt a slab of volume based on verte_id
  tissue_verte: array contains tissue+vertebrate segmentations
  '''
  slice_ids = np.where(tissue_verte==verte_id)[2]
  if len(slice_ids) > 0:
    slice_min,slice_max = max(0,slice_ids.min()), min(slice_ids.max(),tissue_verte.shape[2])
  else:
    slice_min,slice_max = 0,0
  return slice_min, slice_max

#%% 
def extract_verte_slab(tissue_verte, verte_id):
  ''' extraxt a slab of volume based on verte_id
  tissue_verte: array contains tissue+vertebrate segmentations
  '''
  slice_ids = np.where(tissue_verte==verte_id)[2]
  if len(slice_ids) > 0:
    slice_min,slice_max = max(0,slice_ids.min()), min(slice_ids.max(),tissue_verte.shape[2])
    # get the volume slab
    slab = tissue_verte[:,:,slice_min:slice_max]
  else: slab = np.empty(0) # no slice found
  return slab
#%%
def extract_verte_range_slab(label_map, label_id_min, label_id_max):
  ''' extraxt a slab of volume based on verte_id_min, verte_id_max
  label_map: [width, depth, height(slice_through)]
  '''
  slice_ids = np.where((label_map > label_id_min) & (label_map < label_id_max))[2]
  if len(slice_ids) > 0:
    slice_min,slice_max = max(0,slice_ids.min()), min(slice_ids.max(), label_map.shape[2])
    # get the volume slab
    slab = label_map[:,:,slice_min:slice_max]
  else: slab = np.empty(0) # no slice found
  return slab

#%%
def cal_per_verte_tissue_vol(tissue_verte, voxel_size, vol_array=np.empty(0), verte_idx=list(range(303,325)), tissue_idx=[1, 3, 5, 7], verbose=False):
  ''' Calculate tissue per vertebrate
  '''
  vol_array = np.empty(0)
  for tissue_id in tissue_idx:
    if verbose==True: print(f"tissue: {tissue_id}")
   
    for verte_id in verte_idx:
      if verbose==True: print(f"vertebrate: {verte_id}")
      # extract vertebrate slab
      slab = extract_verte_slab(tissue_verte, verte_id)
      
      if len(slab)>0:
        # non-empty slabcal_tissue_vol_per_verte
        # calculate the tissue volume for each vertebrate level
        tissue_vol = cal_label_vol(slab, tissue_id, voxel_size)
      else: # empty slab
        tissue_vol = np.nan
        
      vol_array = np.append(vol_array, tissue_vol)
      # if verbose is True: print(f"tissue vol: {tissue_array}")
    
  return vol_array
#%%
def cal_tissue_vol_per_verte(tissue_verte, voxel_size, vol_array=np.empty(0), verte_idx=list(range(303,325)), tissue_idx=[1, 3, 5, 7], verbose=False):
  ''' Calculate tissue per vertebrate
  '''
  for verte_id in verte_idx:
    if verbose==True: print(f"vertebrate {verte_id}")
    # extract vertebrate slab
    slab = extract_verte_slab(tissue_verte, verte_id)
    if len(slab)>0:
      # non-empty slabcal_tissue_vol_per_verte
      # calculate the tissue volume for each vertebrate level
      vol_array = cal_label_vols_array(slab, tissue_idx, voxel_size, vol_array=vol_array, verbose=verbose)
    else: # empty slab
      tissue_array = np.array([np.nan]*len(tissue_idx))
      vol_array = np.append(vol_array, tissue_array)
      if verbose is True: print(f"tissue vol: {tissue_array}")
    
  return vol_array
#%%
def cal_organ_plus_tissue_vol_verte(organ,tissue_verte, voxel_size, vol_array=np.empty(0),
                                    organ_idx = [4,6,8,9,10,11,12,14,17,20],
                                    verte_idx = list(range(303,325)),
                                    tissue_idx = [1, 3, 5, 7],
                                    verbose=False
                                    ):
  # calculate organ volumes
  vol_array = cal_label_vols_array(organ, organ_idx, voxel_size, vol_array, verbose)
  # calculate tissue-per vertebrate volumes
  #   Separate vertebrate by tissue
  vol_array = cal_per_verte_tissue_vol(tissue_verte, voxel_size, vol_array, verte_idx, tissue_idx, verbose)
  #   Separete tissue by vertebrate
  # vol_array = cal_tissue_vol_per_verte(tissue_verte, voxel_size, vol_array, verte_idx, tissue_idx, verbose)
  return vol_array

#%%
def extract_organ_plus_tissue_vol_verte(organ_path, tissue_verte_path, csv_path, 
                                        organ_idx = [4,6,8,9,10,11,12,14,17,20],
                                        verte_idx = list(range(303,325)),
                                        tissue_idx = [1, 3, 5, 7],
                                        verbose=False, overwrite=False
                                        ):
  # organ_path        = sys.argv[1]
  # tissue_verte_path = sys.argv[2]
  # verbose           = sys.argv[3]
    
  #%%
  if overwrite is False and os.path.isfile(csv_path):
    if verbose is True: print(f"{csv_path} exist, skipping ...")
    return
  else: # delete {csv_path} first before writing
    if os.path.isfile(csv_path): os.remove(csv_path)
  
  #%% calculate tissue volumes
  if os.path.isfile(organ_path):
    if verbose is True: print(f"loading {organ_path}")
    organ_nii = nib.load(f"{organ_path}")
    organ = organ_nii.get_fdata()
    voxel_size = organ_nii.header.get_zooms()
    vol_array_organ = cal_label_vols_array(array=organ, ids=organ_idx, 
                                           voxel_size=voxel_size, verbose=verbose)
  else:
    # return
    # create all zero numpy array
    vol_array_organ = np.array([np.nan]*len(organ_idx))
  #%% calculate vertebrate volumes
  if os.path.isfile(tissue_verte_path):
    if verbose is True: print(f"loading {tissue_verte_path}")
    tissue_verte_nii = nib.load(f"{tissue_verte_path}")
    tissue_verte = tissue_verte_nii.get_fdata()
    voxel_size = tissue_verte_nii.header.get_zooms()
    # cal_tissue_vol_per_verte
    vol_array_tissue_verte = cal_per_verte_tissue_vol(tissue_verte=tissue_verte, 
                                                      voxel_size=voxel_size, 
                                                      verte_idx=verte_idx, 
                                                      tissue_idx=tissue_idx,
                                                      verbose=verbose)
  else:
    # return
    vol_array_tissue_verte = np.array([np.nan]*(len(tissue_idx)*len(verte_idx)))
  #%% concatenate two vol_arrays
  vol_array = np.append(vol_array_organ, vol_array_tissue_verte)

  
  #%% [old implementation] calculate all volumes
  # vol_array = cal_organ_plus_tissue_vol_verte(organ, tissue_verte, voxel_size, verbose)
  
  #%% save vol_array into a csv
  np.savetxt(csv_path, vol_array, delimiter=",")
  #%%
  return vol_array
  
#%%
