# %%
import os
from glob import glob
# %% 
# data root
os.environ["DATA_ROOT"] = f"/project/rrg-mfbeg-ad/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/UBCMIXDEM_WMHT1T2relationships"

data_root = os.environ["DATA_ROOT"]
# yaml config file
yaml_file = f"_wm_segs/_wm_segs.yaml"
# checkpoint path
ckpt_path = glob(f"{data_root}/PROCESSED_DATA/checkpoints/*.ckpt")[0]

input_paths = glob(f"{data_root}/RAW_DATA/*_T1W.nii.gz")[-1]
output_paths = ''
# %%
