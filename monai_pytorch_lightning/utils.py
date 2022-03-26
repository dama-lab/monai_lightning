#%% import libraries
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import yaml
from monai.data import NiftiSaver
# from torch.utils.data  import Dataset, DataLoader
from monai.data import Dataset, DataLoader
# from monai.config import DtypeLike, KeysCollection, NdarrayTensor
# import monai.transforms as t
# from monai.transforms.spatial.dictionary import *

def init_setup():
  '''
  some boilplate code
  # Test suit:
  # init_setup()
  '''
  # import os, sys
  # code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline"
  # sys.path.insert(1,code_dir)
  # os.chdir(code_dir)

  usr = os.getenv("USER")
  tmp_dir = f'/tmp/{usr}'
  Path(tmp_dir).mkdir(exist_ok=True)
  return tmp_dir


#%%=======================================
#          File reading
#=========================================
#%% define yaml reading function
def read_yaml(yaml_file):
  '''read yaml file
  # Test suit:
  # yaml_files = ''
  # read_yaml(yaml_file)
  '''  
  try:
    with open(yaml_file, 'r') as stream:
      try:
          yaml_dict = yaml.safe_load(stream)
          return yaml_dict
      except yaml.YAMLError as exc:
          print(exc)
  except FileNotFoundError as e:
    print("Error reading the config file {yaml_file}")
    print(e)

def read_nib(input_file, return_type="numpy"):
  '''read nibabel supported file format
  return_type: numpy / nibabel
  '''
  import nibabel as nib
  nib_obj = nib.load(input_file)
  if return_type == "nibabel":
    return nib_obj
  elif return_type == "numpy":
    vol = nib_obj.get_fdata().transpose()
    return vol 

def save_batch_to_nifti(batch_vol, meta_data=None, output_dir='./', output_postfix='', output_ext='.nii.gz', resample=False, squeeze_end_dims=True, data_root_dir='', separate_folder=False, print_log=True):
  '''save vol (torch.tensor or numpy.Array) to nibabel supported file format'''
  # %%
  nifti_saver = NiftiSaver(output_dir=output_dir, output_postfix=output_postfix, output_ext=output_ext, resample=resample, squeeze_end_dims=squeeze_end_dims, data_root_dir=data_root_dir, separate_folder=separate_folder, print_log=print_log)

  nifti_saver.save_batch(batch_data=batch_vol, meta_data=meta_data)

# =================================================================
#%% =========== file conversion ======= (e.g. convert mgz to nifti)
# =================================================================

def mriconvert(input_file, output_file, output_type, return_type=None, exe=True):
  '''Ref：https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.freesurfer.preprocess.html#mriconvert
  - file type: (‘cor’ or ‘mgh’ or ‘mgz’ or ‘minc’ or ‘analyze’ or ‘analyze4d’ or ‘spm’ or ‘afni’ or ‘brik’ or ‘bshort’ or ‘bfloat’ or ‘sdt’ or ‘outline’ or ‘otl’ or ‘gdf’ or ‘nifti1’ or ‘nii’ or ‘niigz’ or ‘ge’ or ‘gelx’ or ‘lx’ or ‘ximg’ or ‘siemens’ or ‘dicom’ or ‘siemens_dicom’)
  - return_type: "cmdline"/
  '''
  from nipype.interfaces.freesurfer.preprocess import MRIConvert
  mc = MRIConvert()
  mc.inputs.in_file = input_file
  mc.inputs.out_file = output_file
  mc.inputs.out_type = output_type
  if exe == True:
    mc.run()
  if return_type == "cmdline":
    return mc.cmdline
  else:
    return mc

#%%=======================================
#                 Analysis
#=========================================

def cal_dice(source, target, label=1):
  '''dice for two label array for label=label'''
  import torch
  source = torch.tensor(source, dtype=torch.int64)
  target = torch.tensor(target, dtype=torch.int64)
  # put both into two 1d array (same as torch.flatten()) (ref: https://github.com/cezannec/capsule_net_pytorch/issues/4)
  source = (source==label).long().contiguous().view(-1)
  target = (target==label).long().contiguous().view(-1)
  
  intersect = (source * target).sum(dim=0).float()
  union = (source + target).sum(dim=0).float()
  
  dice = 2. * intersect / union
  
  return dice

# =============== poke/peek network architecture details ====================
#%% check parameter size
def print_param_size(net):
  # check parameter size
  print("=== parameter size ===")
  for p in net.parameters():
      print(list(p.shape))

#%% check network layerwise feature size
def print_layerwise_feature_size(net, X, batch_size=50):
    # from IPython.display import display, Markdown, Latex
    print("=== feature size ===")
    if isinstance(X, Dataset):
      X = DataLoader(X, batch_size=batch_size)
    if isinstance(X, DataLoader):
      X, _ = next(iter(X))
    if isinstance(X, torch.Tensor) and len(X.shape) < 1:
      X = x.reshape(1,-1)
    for layer in net:
        X = layer(X)
        print(f"{layer.__class__.__name__:<8} \t output shape: \t {list(X.shape)}")
        # display(Markdown(f"**{layer.__class__.__name__}** output shape: \t {X.shape}"))
