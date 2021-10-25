#%% import libraries
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import yaml
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