#%% import libraries
import os, sys, torch
from pathlib import Path
from importlib import reload
import monai
from med_deeplearning.HelperFunctions import visualization as v

code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline/lightning_library"
sys.path.insert(1,code_dir)
os.chdir(code_dir)

import datasets, train, utils, models as m
from utils import LabelValueScaled

# %% defining data locations
root_dir = Path("/project/6003102/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/")
processed_dir = root_dir/"PROCESSED_DATA"
processed_dir.mkdir(exist_ok=True, parents=True)
images_dir = Path(f"{root_dir}/RAW_DATA/")
# T1
image_paths_t1 = sorted(images_dir.glob("*_T1W.nii.gz"))
# T2
image_paths_t2 = sorted(images_dir.glob("*_T2WFLAIRinT1W.nii.gz"))
# T1 label
label_paths_t1 = sorted(images_dir.glob("*_T1HYPOWMSAinT1W.nii.gz"))
# T2 label
label_paths_t2 = sorted(images_dir.glob("*_T2HYPERWMSAinT1W.nii.gz"))
# WM label
label_path_WM  = sorted(images_dir.glob("*_WMPARCinT1W.nii.gz"))
image_paths, label_paths = image_paths_t1, label_paths_t2
assert len(image_paths) == len(label_paths)

#%% create DataModule
reload(datasets)
data_wmh = datasets.DataModuleWMH(image_paths_t1, label_paths_t2, label_transformations = [LabelValueScaled(keys=['label'],scale=1/1000)], batch_size=15, dataset_type="CacheDataset")
data_wmh.setup()

#%% create model (LightningModule)
reload(m)
unet_wmh = m.UNetWMH(m.unet_3d(), loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True), optimizer_class=torch.optim.AdamW, lr=1e-3) # lr=50, 1e-2
print(unet_wmh.lr)

#%% define trainer
reload(train)
exp_name="UBC_UNet_DiceCE_lr_0.01"
trainer = train.lightning_trainer(default_root_dir=processed_dir, log_dir = f"{processed_dir}/logs/{exp_name}", max_epochs=600, exp_name=exp_name)

# #%% Start training with learning rate = 1e-3
# trainer.fit(unet_wmh, data_wmh)

# #%% save the model
# checkpoint_save_path = f"{trainer.default_root_dir}/UBC_UNet_DiceCE_best_model.ckpt"
# trainer.save_checkpoint(checkpoint_save_path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# test / evaluate
#%% load validation batch
b = 0
val_batch = next(iter(data_wmh.val_dataloader()))
img_b, lbl_b = val_batch['image'], val_batch['label']
img, lbl = img_b[b,0,...], lbl_b[b,0,...]
img.shape, lbl.shape
#%% load train batch
b = 0
train_batch = next(iter(data_wmh.train_dataloader()))
img_b, lbl_b = train_batch['image'], train_batch['label']
img, lbl = img_b[b,0,...], lbl_b[b,0,...]
img.shape, lbl.shape
#%% alternatively: (doesn't have channel dimention though)
# idx = 0
# val_batch = data_wmh.valid_data[idx]
# img, lbl = val_batch['image'], val_batch['label']
# img, lbl = img[0,...], lbl[0,...]

#%% load/plot validation batch
reload(utils)
img, lbl = utils.load_plot_batch(data_wmh.val_dataloader)

#%% load/plot training batch
reload(utils)
img, lbl = utils.load_plot_batch(data_wmh.train_dataloader)

#%%%%%%%%% evaluate by inference the model
#%% load the saved checkpoint
checkpoint_save_path = f"{trainer.default_root_dir}/UBC_UNet_DiceCE_best_model.ckpt"
m = unet_wmh.load_from_checkpoint(checkpoint_save_path);

#%% inference
pred_b = unet_wmh.model.forward(img_b.to('cuda'))
pred = pred_b[b,...].argmax(dim=0)
#%% find the min bbox containing White-Matter-Leision
reload(utils)
bb = utils.get_label_bbox(lbl)

# %% %%%% plot batch data
reload(utils)
img_crop = utils.crop_img_bbox(img,bb).detach()
lbl_crop = utils.crop_img_bbox(lbl,bb).detach()

v.vol_peek(vol=img_crop, overlay_vol=lbl_crop)
# %%
pred_crop = utils.crop_img_bbox(pred,bb).detach().cpu()
v.vol_peek(vol=img_crop, overlay_vol=pred_crop)

# %%
