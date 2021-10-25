import os
code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline/lightning_library"
sys.path.insert(1,code_dir)
os.chdir(code_dir)
from lightning_library import utils
from importlib import reload
import numpy as np

#%% convert mgz into nii
reload(utils)
input_file = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/FREESURFER/MIXDS0001_v0/mri/nu.mgz"
output_file = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/nifti/MIXDS0001_v0/nu.nii"
m = utils.mriconvert(input_file, output_file, 'niigz')

#%% Conclusion: Hyunwoo's images are in subject's original space
reload(utils)
input_file = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/FREESURFER/MIXDS0001_v0/mri/orig_nu.mgz"
output_file = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/nifti/MIXDS0001_v0/orig_nu.nii"
m = utils.mriconvert(input_file, output_file, 'niigz')

#%% check WM Seg label values:
reload(utils)
wmseg_files= "/project/6003102/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/RAW_DATA/UBCMIXDEM_MIXDS0001_v0_WMPARCinT1W.nii.gz"
vol = utils.read_nib(wmseg_files)
np.unique(vol) # [ 0.,  5., 11., 12., 13., 14., 21., 22., 23., 24.]

#%% ============== pilot test ==============
#%% === test suit for `LabelValueRemapd` ===
#%% ========================================
from lightning_library import transforms as t
import numpy as np
reload(t)
img_in = np.array([[1,2],[3,4],[5,6]])
img_in
## test label_value_remap
t.label_value_remap(img_in,[2,5],[7,8], True)
## test LabelValueRemap
# Case 1: not zero remaining labels
trans = t.LabelValueRemap([2,5],[7,8], False) # default: True
trans(img_in)
# Case 2: zero remaining labels
trans = t.LabelValueRemap([2,5],[7,8]) # default: True
trans(img_in)
## test LabelValueRemapd
transd = t.LabelValueRemapd(keys="img", old_labels=[2,5], new_labels=[7,8])
transd({"img": img_in})

