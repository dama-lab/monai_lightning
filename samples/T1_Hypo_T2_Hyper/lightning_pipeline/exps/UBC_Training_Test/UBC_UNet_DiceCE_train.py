# Reference: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
# %% setup exe mode: local/cedar/google
exe_mode = "local_linux" # "cedar" # "local_windows"

# %% import libraries
print('  >>> importing libraries...')
import os, sys, torch
from pathlib import Path
from importlib import reload
import numpy as np

import wandb
import pytorch_lightning as pl

import monai
from monai.data import decollate_batch

import sys
if exe_mode == "cedar": # remote on cedar
  code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline/lightning_library"
elif exe_mode == "local_linux": # wls
  code_dir = "/mnt/c/Users/madam/OneDrive/Codes/VHA/vha-dnn/med_deeplearning/med_deeplearning/monai_pytorch_lightning"
elif exe_mode == "local_windows": # local
  code_dir = r"C:\Users\madam\OneDrive\Codes\VHA\vha-dnn\med_deeplearning\cohorts\BrainMRI\T1_Hypo_T2_Hyper\lightning_pipeline\lightning_library"
  

sys.path.insert(1,code_dir)
# os.chdir(code_dir)

#%% from lightning_library 
import datasets, train, utils, models
from transforms import LabelValueScaled, LabelValueRemapd
import visualization as v

# %% defining data locations
print('  >>> defining data locations...')
if exe_mode == "cedar": # remote on cedar
  data_root = Path("/project/rrg-mfbeg-ad/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA")
elif exe_mode == "local_linux": # local
  data_root = Path("/mnt/c/Users/madam/OneDrive/Data/Brain_MRI/T1_Hypo_T2_Hyper/")
elif exe_mode == "local_windows": # local
  data_root = Path("C:\\Users\madam\OneDrive\Data\Brain_MRI\T1_Hypo_T2_Hyper")
  # or: Path("C:/Users/madam/OneDrive/Data/Brain_MRI/T1_Hypo_T2_Hyper")

#%%
root_dir = data_root/"UBCMIXDEM_WMHT1T2relationships"  
  
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
label_paths_wm  = sorted(images_dir.glob("*_WMPARCinT1W.nii.gz"))
image_paths, label_paths = image_paths_t1, label_paths_t2
assert len(image_paths) > 0 # ensure there are files
assert len(image_paths) == len(label_paths)


#%% exp0: create DataModule for T1 WM Tissue Seg label
print('  >>> creating DataModule...')
reload(datasets)
batch_size = 1
old_labels = [5, 11, 12, 13, 14, 21, 22, 23, 24]
new_labels = list(range(1,len(old_labels)+1))
label_transformations = [LabelValueRemapd(keys=['label'], old_labels=old_labels, new_labels=new_labels, zero_other_labels=True)]
data_wms = datasets.DataModuleWMH(image_paths_t1, label_paths_wm, patch_size=(64, 64, 64), label_transformations=label_transformations, batch_size_train=batch_size, dataset_type="CacheDataset", num_workers=0)
data_wms.setup()


# %% look at the dataset
preview = False
if preview == True:
  reload(utils)
  reload(monai)
  iterator = iter(data_wms.train_dataloader())
  torch.set_printoptions(precision=4)
  for i in range(len(data_wms.train_files)):
  #     sample_data = data_wms.train_files[i]
  #     img_path, lbl_path = sample_data['image'], sample_data['label']
  #     img, lbl = utils.read_nib(img_path), utils.read_nib(lbl_path)

      val_batch = next(iterator)
      img_b, lbl_b = val_batch['image'].type('torch.FloatTensor'), val_batch['label'].type('torch.FloatTensor')
      print(img_b.min(), img_b.mean(), img_b.std(), img_b.max())
  #     print(lbl_b.shape)
  #     print(lbl_b.unique())
  #     break


# #  preview
# img, lbl = img_b[b,0,...], lbl_b[b,0,...]
# fig = v.vol_peek(vol=img, overlay_vol=lbl)

# fig,_,_ = utils.load_plot_batch(data_wms.train_dataloader,crop=False, dilate=10)
# fig_path = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/PROCESSED_DATA/tmp/preview_wmseg_overlay.png"
# fig.savefig(fig_path)


# %% create model
reload(models)
lr=1e-3
label_num = 10 # = len(lbl_b.unique())
device = 'cuda'
net = models.unet_3d(in_channels=1, out_channels=label_num, num_res_units=0)
unet_wms = models.UNet3DModule(net=net, lr=lr, device=device)
print(unet_wms.lr)

#%% test model prediction
# load previously-trained model
if exe_mode == "local_windows":
  checkpoint_path = r"C:\Users\madam\OneDrive\Data\Brain_MRI\T1_Hypo_T2_Hyper\RAW_DATA\UBCMIXDEM_WMHT1T2relationships\PROCESSED_DATA\checkpoints\UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_0.001-epoch=9_val_loss=2.8229_val_dice=0.0169.ckpt"
else:
  checkpoint_path = data_root/"PROCESSED_DATA/models/UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_0.001-epoch=9_val_loss=2.8229_val_dice=0.0169.ckpt" 
#"Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/data/PROCESSED_DATA/models/UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_0.001-epoch=9_val_loss=2.8229_val_dice=0.0169.ckpt"
# unet_wms.model = unet_wms._model
# unet_wms.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net);
unet_wms.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net, strict=False);
#%%
iterator = iter(data_wms.train_dataloader())
val_batch = next(iterator)
#%%
img_b, lbl_b = val_batch['image'], val_batch['label']
unet_wms.to('cuda');
preds = unet_wms(img_b.to('cuda'))
preds_argmax = preds.argmax(axis=1)
#%%
b = 3
pred  = preds_argmax[b,...]
print("prediction ...")
fig = v.vol_peek(vol=img_b[b,0,...], overlay_vol=pred.detach().cpu())
print("ground truth ...")
fig = v.vol_peek(vol=img_b[b,0,...], overlay_vol=lbl_b[b,0,...])
#%%
for lbl_id in range(1,9):
  print(lbl_id,"ground truth ...")
  fig = v.vol_peek(vol=img_b[b,0,...], overlay_vol=lbl_b[b,0,...]==lbl_id)
  print(lbl_id,"prediction ...")
  fig = v.vol_peek(vol=img_b[b,0,...], overlay_vol=pred.detach().cpu()==lbl_id)

#%%%%% All the 3 types 
#%% compute dice use my own function
# using my own dice calculation function (the results appears to be the same)
reload(utils)
num_classes = 10
dice_array = torch.zeros(lbl_b.shape[0],num_classes)
for b in range(lbl_b.shape[0]):
  for lbl_id in range(1,num_classes):
    pred = preds_argmax[b,...].detach().cpu()
    lbl = lbl_b[b,0,...]
    dice = utils.cal_dice(pred, lbl, lbl_id)
    dice_array[b,lbl_id] = dice
print(dice_array[...,1:])
  # utils.cal_dice(lbl_b[b,0,...], pred.type(torch.FloatTensor).detach().cpu(), 6)

#%% == Convert label to onehot format
# - apporach 1: use pytorch's onehot function （by default, auto-determine number of classes, risky）
 # Ref: https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb
# - have to be int64: e.g. `lbl_b.to(torch.int64).dtype` (`lbl_b.type(torch.IntTensor).dtype` doesn't work)
# - dim0=squeeze: remove the channel dimension (only one label channel is available)
# - permute onehotencode to channel 1 i.e. from [4, 64, 64, 64, 10] => [4, 10, 64, 64, 64]
# preds.shape=[4,10, 64,64,64]; lbl_b.shape=[4, 1, 64, 64, 64]
num_classes = 10
lbl_onehot = torch.nn.functional.one_hot(lbl_b[:,0,...].to(torch.int64), num_classes=num_classes).permute(0,4,1,2,3)
preds_onehot = torch.nn.functional.one_hot(preds_argmax, num_classes=num_classes).permute(0,4,1,2,3)
# - approach 2: use MONAI's `monai.networks.utils.one_hot` function 
#   (Ref: https://docs.monai.io/en/0.1.0/apidocs/monai.networks.utils.html)
lbl_onehot_monai = monai.networks.utils.one_hot(lbl_b,num_classes=10)
assert((lbl_onehot - lbl_onehot_monai).sum()==0)

# naive way of using monai.metrics.DiceMetric directly on one-hot-encoded tensors
get_dice = monai.metrics.DiceMetric(include_background=False,reduction='none',get_not_nans=True) 
dices = get_dice(y_pred=preds_onehot,y=lbl_onehot.to(device))
dices

#%%%%%%%%%%%%%%%%
#%% evaluate using DiceMetrics
# decollate batch use MONAI's 'decollation_batch' function
# Ref: https://colab.research.google.com/github/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb#scrollTo=d4zasnv3wOXU
lbl_b_list = decollate_batch(lbl_b) # lbl_b or lbl_onehot?
preds_list = decollate_batch(preds)
# check decollated list shape [1, 64, 64, 64] or [10, 64, 64, 64]
[i.shape for i in lbl_b_list]
[i.shape for i in preds_list]

outputs = [unet_wms.post_label(i).to(device) for i in decollate_batch(lbl_b)]
labels  = [unet_wms.post_pred(i).to(device) for i in decollate_batch(preds)]

[i.shape for i in outputs]
[i.shape for i in labels]


#%% compute dice on a list of decollated tensors
# Ref: https://colab.research.google.com/github/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb#scrollTo=d4zasnv3wOXU
# reduction='mean',get_not_nans=False
get_dice = monai.metrics.DiceMetric(include_background=False,reduction='mean',get_not_nans=False)
dices = get_dice(y_pred=outputs,y=labels)
dices
#%% reduction='mean', get_not_nans=False
get_dice = monai.metrics.DiceMetric(include_background=False,reduction='mean',get_not_nans=True)
dices = get_dice(y_pred=outputs,y=labels)
dices
#%% get_not_nans=False will get a second returned value
get_dice = monai.metrics.DiceMetric(include_background=False,reduction='none',get_not_nans=True) 
dices = get_dice(y_pred=outputs,y=labels)
dices


#%% define trainer
reload(train)
exp_name = f"UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_{lr}"
trainer = train.lightning_trainer(default_root_dir=processed_dir, log_dir = f"{processed_dir}/logs/{exp_name}", max_epochs=600, exp_name=exp_name, monitor="val_dice")

#%% unet_wms.to('cpu');
lr_finder = trainer.tuner.lr_find(model=unet_wms, datamodule=data_wms,min_lr = 1e-5, max_lr = 1e2, num_training = 100)

#%% Start training with learning rate = {lr}
print(f'  >>> Start training with learning rate = {lr}...')
trainer.fit(unet_wms, data_wms)

#%% save the model
print('  >>> save the model...')
checkpoint_save_path = f"{trainer.default_root_dir}/UBC_UNet_WMS_DiceCE_best_model_Dice_0.5146.ckpt"
trainer.save_checkpoint(checkpoint_save_path)


#%% ========================================
#%% exp1: create DataModule for T2 WMH label
print('  >>> creating DataModule...')
reload(datasets)
data_wms = datasets.DataModuleWMH(image_paths_t1, label_paths_t2, patch_size=(96, 96, 96), label_transformations = [LabelValueScaled(keys=['label'],scale=1/1000)], batch_size=15, dataset_type="CacheDataset")
data_wmh.setup()

#%% create model (LightningModule)
print('  >>> creating model (LightningModule)...')
reload(m)
lr=1e-3 # learning rate
unet_wmh = models.UNetWMH(models.unet_3d(), loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True), optimizer_class=torch.optim.AdamW, lr=lr)
print(unet_wmh.lr)

#%% define trainer
print('  >>> defining trainer...')
reload(train)
exp_name=f"UBC_UNet_DiceCE_lr_{lr}"
trainer = train.lightning_trainer(default_root_dir=processed_dir, log_dir = f"{processed_dir}/logs/{exp_name}", max_epochs=600, exp_name=exp_name, monitor="val_dice")

#%% Start training with learning rate = {lr}
print(f'  >>> Start training with learning rate = {lr}...')
trainer.fit(unet_wmh, data_wmh)

#%% save the model
print('  >>> save the model...')
checkpoint_save_path = f"{trainer.default_root_dir}/UBC_UNet_DiceCE_best_model.ckpt"
trainer.save_checkpoint(checkpoint_save_path)

#%% %%%%% debug %%%%%%%%%%
pred = unet_wms(img_b).argmax(axis=1)[0,...]
x, y = img_b, lbl_b
y_hat = unet_wms.forward(x)
loss = unet_wms.loss_function(y_hat, y)
valid_dice = models.cal_mean_dice(y_hat, y)

# implemented in library function
batch = val_batch
  def validation_step(self, batch, batch_idx):
      dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
      images, labels = batch["image"], batch["label"]
      roi_size = (96, 96, 96)
      sw_batch_size = 4
      outputs = sliding_window_inference(
          images, roi_size, sw_batch_size, predictor=unet_wms.forward)
      loss = unet_wms.loss_function(outputs, labels)
      
      outputs = [unet_wms.post_pred(i) for i in decollate_batch(outputs)]
      labels = [unet_wms.post_label(i) for i in decollate_batch(labels)]
      valid_dice = dice_metric(y_pred=outputs, y=labels)
