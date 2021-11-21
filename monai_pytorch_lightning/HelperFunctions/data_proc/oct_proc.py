# Ref: 
# %% import library
import numpy as np

# %%
def surf2label(surf, vol, label_type="patch", axial=0, lateral=1, depth=2, verbose=False):
  '''convert surf to label (layer)
  surf and vol must have the same dimension orientation (see below):
  - surf: H(L)xWxD  (axial,lateral,depth = 0,1,2)
  - vol :    HxWxD  (axial,lateral,depth = 0,1,2)
  - label_type = "patch", "boundary"
  # x,y,z: [0]=axial, [1]=lateral, [2]=depth 
  axial,lateral,depth = 0,1,2 # x,y,z
  '''
  # %%
  # initialize label volume    
  label = np.uint16(np.zeros(vol.shape))

  for layer_id in range(surf.shape[axial]):
    surf_cur = surf[layer_id,:,:] ##(lateral, depth)
    # x,y,z = depth,lateral,axial
    # np.meshgrid take lateral as the 1st axis, axial as the 2nd axis
    _,axial_gradiant,_ = np.meshgrid(np.arange(vol.shape[lateral])+1,
                            np.arange(vol.shape[axial])+1,
                            np.arange(vol.shape[depth])+1)
    #%%
    surf_repmat = np.tile(surf_cur,[vol.shape[axial],1,1])
    if label_type == "boundary":
      layer_cur = (layer_id+1) * (axial_gradiant == surf_repmat)
      label = label*(layer_cur==0) + layer_cur
    elif label_type == "patch":
      label = label + np.uint16(axial_gradiant > surf_repmat)
    # break 
  return label

def vol2surf(vol):
  volSeg = np.array(vol)
  layerIds = np.unique(volSeg)
  surfs = np.empty([0,*(volSeg.shape[1:])])
  for layer in layerIds:
    # skip background (label == 0 or np.nan)
    if (layer == 0) or (layer == np.nan): continue
    surf = np.expand_dims(np.argmax(volSeg==layer-1,axis=0),axis=0)
    # if lower surf become above the higher surf, put it one voxel below the above
    if 'surf_previous' in vars():
      swap_idx = (surf<=surf_previous)
      surf[swap_idx] = surf_previous[swap_idx]+1
    surfs = np.concatenate([surfs,surf],axis=0) if surfs.size else surf
    surf_previous = surf
  return surfs

# %%
def boundary2surf(boundary):
  '''
  convert one-voxel thick boundary-to-surface
  '''
  # %% determine array of layer number
  layers_array = np.unique(boundary)
  # %% remove background layers (=0)
  layers_array = layers_array(layers_array != 0)
  # %% generate surf 
  surf = np.zeros(len(layers_array), boundary.shape[1], boundary.shape[2])
  for idx in range(len(layers_array)):
    boundary_cur = (boundary == layers_array(idx))
    surf[idx,:,:] = np.argmax(boundary_cur, axis=0)
  
  return surf
        
  

