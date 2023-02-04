exe_mode = "local_windows"
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% plot loss v.s. epochs 
if exe_mode == "local_windows":
  train_log_path = r"C:\Users\madam\OneDrive\Data\Brain_MRI\T1_Hypo_T2_Hyper\RAW_DATA\UBCMIXDEM_WMHT1T2relationships\PROCESSED_DATA\logs\UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_0.001\csv_log\version_5\metrics.csv"
  
train_log_df = pd.read_csv(train_log_path)
train_log = train_log_df.val_dice.to_numpy()
train_log = train_log[np.logical_not(np.isnan(train_log))]
plt.plot(train_log)


#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% visualize segmentation using trained models

# %% import libraries
print('  >>> importing libraries...')
import os, sys, torch
from pathlib import Path
from importlib import reload
import numpy as np
import monai

if exe_mode == "cedar": # remote on cedar
  code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline/lightning_library"
elif exe_mode == "local_windows": # local
  code_dir = r"C:\Users\madam\OneDrive\Codes\VHA\vha-dnn\med_deeplearning\cohorts\BrainMRI\T1_Hypo_T2_Hyper\lightning_pipeline\lightning_library"

sys.path.insert(1,code_dir)
os.chdir(code_dir)

# from lightning_library 
import datasets, train, utils, models
from transforms import LabelValueScaled, LabelValueRemapd
import visualization as v

# %% defining data locations
print('  >>> defining data locations...')
if exe_mode == "cedar": # remote on cedar
  root_dir = Path("/project/6003102/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/")
elif exe_mode == "local_windows": # local
  root_dir = Path(r"C:\Users\madam\OneDrive\Data\Brain_MRI\T1_Hypo_T2_Hyper\RAW_DATA\UBCMIXDEM_WMHT1T2relationships")

processed_dir = root_dir/"PROCESSED_DATA"
processed_dir.mkdir(exist_ok=True, parents=True)
images_dir = Path(f"{root_dir}/RAW_DATA/")
# T1
image_paths_t1 = sorted(images_dir.glob("*_T1W.nii.gz"))
# T2
image_paths_t2 = sorted(images_dir.glob("*_T2WFLAIRinT1W.nii.gz"))
# T1 label
label_paths_t1 = sorted(images_dir.glob("*_T1HYPOWMSAinT1W.nii.gz"))
# T2 label (need to be devided by 1000)
label_paths_t2 = sorted(images_dir.glob("*_T2HYPERWMSAinT1W.nii.gz"))
# WM label
label_path_WM  = sorted(images_dir.glob("*_WMPARCinT1W.nii.gz"))
image_paths, label_paths = image_paths_t1, label_paths_t2
assert len(image_paths) > 0 # ensure there are files
assert len(image_paths) == len(label_paths)


#%% exp2: create DataModule for T1 WM Tissue Seg label
print('  >>> creating DataModule...')
reload(datasets)
batch_size = 1
old_labels = [5, 11, 12, 13, 14, 21, 22, 23, 24]
new_labels = list(range(1,len(old_labels)+1))
label_transformations = [LabelValueRemapd(keys=['label'], old_labels=old_labels, new_labels=new_labels, zero_other_labels=True)]
data_wms = datasets.DataModuleWMH(image_paths_t1, label_path_WM, patch_size=(64, 64, 64), label_transformations=label_transformations, batch_size=batch_size, dataset_type="CacheDataset", num_workers=0)
data_wms.setup()

## create model
reload(models)
lr=1e-3
label_num = 10 # = len(lbl_b.unique())
device = 'cpu'
net = models.unet_3d(in_channels=1, out_channels=label_num, num_res_units=0)
unet_wms = models.UNetWMH(net=net, lr=lr, device=device)
print(unet_wms.lr)

#%% test model prediction
# load previously-trained model
if exe_mode == "local_windows":
  checkpoint_path = r"C:\Users\madam\OneDrive\Data\Brain_MRI\T1_Hypo_T2_Hyper\RAW_DATA\UBCMIXDEM_WMHT1T2relationships\PROCESSED_DATA\checkpoints\UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_0.001-epoch=9_val_loss=2.8229_val_dice=0.0169.ckpt"
  unet_wms.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net);

iterator = iter(data_wms.train_dataloader())
val_batch = next(iterator)
img_b, lbl_b = val_batch['image'], val_batch['label']
unet_wms.to('cuda');
pred = unet_wms(img_b.to('cuda')).argmax(axis=1)[0,...]
fig = v.vol_peek(vol=img_b[0,0,...], overlay_vol=pred.cpu())
fig = v.vol_peek(vol=img_b[0,0,...], overlay_vol=lbl_b[0,0,...])

#%% Plot training lot
train_log_path = "/home/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/logs/UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_1e-3/csv_log/version_2/metrics.csv"

import pandas as pd
train_log = pd.read_csv(train_log_path)
# %%
import matplotlib.pyplot as plt
plt.plot(train_log.val_dice)

# %%
