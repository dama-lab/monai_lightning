#%%
# from med_deeplearning.cohorts.BrainMRI.T1_Hypo_T2_Hyper.lightning_pipeline.dataset import DataModuleWMH
# from med_deeplearning.models.losses import dice_loss
# from fastai.torch_core import one_hot
# from fastai.vision.core import shape
# from lightning_library.train import train_lightning
# from med_deeplearning.models import model_evaluation
# from monai.metrics.meandice import DiceMetric
# from operator import gt
from pathlib import Path
import monai
# from monai.data import CacheDataset, Dataset, PersistentDataset
# from monai.utils import set_determinism
# import monai.transforms as t
import pytorch_lightning as pl
# from med_deeplearning.HelperFunctions import visualization as v
#%% import libraries
import os, sys
from torch.autograd.grad_mode import F
code_dir = "/home/dma73/Code/medical_image_analysis/cohorts/BrainMRI/T1_Hypo_T2_Hyper/lightning_pipeline/lightning_library"
sys.path.insert(1,code_dir)
# os.chdir(code_dir)
from lightning_library import utils
from lightning_library.transforms import LabelValueScaled, LabelValueRemapd
#%% defining data locations
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
from lightning_library import datasets
from importlib import reload
reload(datasets)
data_wmh = datasets.DataModuleWMH(image_paths_t1, label_paths_t2, label_transformations = [LabelValueScaled(keys=['label'],scale=1/1000)], batch_size=5)
data_wmh.setup()
#%% create model (LightningModule)
import models as m, torch
from importlib import reload
reload(m)
unet_wmh = m.UNetWMH(m.unet_3d(), loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True), optimizer_class=torch.optim.AdamW, lr=0.5) # lr=50, 1e-2
# loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)
print(unet_wmh.lr)
#%% define trainer
import train
reload(train)
default_root_dir=processed_dir
trainer = train.train_lightning(default_root_dir=default_root_dir,max_epochs=600) # log_every_n_steps=5, flush_logs_every_n_steps=10)

#%% learning rate finder
lr_finder = trainer.tuner.lr_find(model=unet_wmh,datamodule=data_wmh,min_lr = 1e-3, max_lr = 1e4, num_training = 100)
print(unet_wmh.lr)
#%% plot find the best learning rate
lr_finder.plot()
lr = lr_finder.suggestion()
lr, unet_wmh.lr
#%% Start training with learning rate = 1
unet_wmh.lr = 0.5
trainer.fit(unet_wmh, data_wmh)
#%% save the model
checkpoint_save_path = f"{trainer.default_root_dir}/best_model.ckpt"
trainer.save_checkpoint(checkpoint_save_path)
#%% to load model:
unet_wmh.load_state_dict(checkpoint_save_path)


#%% look at inference results
from monai.inferers import sliding_window_inference
net = unet_wmh
n = net.eval();
device = torch.device("cuda:0")
n = net.to(device);
with torch.no_grad():
    for i, val_data in enumerate(net.val_dataloader()):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, net
        )
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(
            val_outputs, dim=1).detach().cpu()[0, :, :, 80])
        plt.show()



#%% check monitored metrics
trainer.logged_metrics
#%% predictinon/inference
# > ref: https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference
# === load model
model_path = trainer.checkpoint_callback.best_model_path
# or: model_path = checkpoint_save_path
pretrained_model = unet_wmh.load_from_checkpoint(checkpoint_path=model_path).to('cuda')
# freeze pretrained model
pretrained_model.eval()
pretrained_model.freeze()

#%%
for val_batch in data_wmh.val_dataloader():
  # option 1
  y_pred = pretrained_model.forward(val_batch['image'].to(pretrained_model.device))
  # option 2
  y_pred_2 = pretrained_model(val_batch['image'].to(pretrained_model.device))
  # check / confirm they're the same
  (y_pred - y_pred_2).norm()
  break
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# self.train_data, self.valid_data = random_split(self.data_dicts, [15, 2])

#%% define logger:
tb_logger = pl.loggers.TensorBoardLogger(save_dir=f"{os.getcwd()}/logs")
csvlogger = pl.loggers.CSVLogger(save_dir=f"{os.getcwd()}/logs")

#%% define trainer
from callbacks import PrintingCallbacks

trainer = pl.Trainer(gpus=1, auto_lr_find=True, accelerator='dp',max_epochs=600, logger=[csvlogger, tb_logger], callbacks=[PrintingCallbacks()],
log_every_n_steps=50,
flush_logs_every_n_steps=100,

#%% check patch sampler size
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for i,i_d in enumerate(data_wmh.train_data):
  for j, j_d in enumerate(i_d):
    print(i,j,len(data_wmh.train_data),len(i_d),j_d['image'].shape, j_d['label'].shape, j_d['label'].unique())
    # break
  # break
#%% check train_dataload batch size
import monai, numpy
from monai.transforms import AsDiscrete

loss_function=monai.losses.DiceLoss(to_onehot_y=True, softmax=True)

for i,batch in enumerate(data_wmh.train_dataloader()):
  images, labels = batch["image"].to('cpu'), batch["label"].to('cpu')
  print(images.shape, labels.shape, labels.unique())
  break
#%%
# y_hat = unet_wmh.forward(images.cuda())
# y_hat.shape

#%% check val_dataload batch size
import monai, numpy
from monai.transforms import AsDiscrete

loss_function=monai.losses.DiceLoss(to_onehot_y=True, softmax=True)

for i,batch in enumerate(data_wmh.val_dataloader()):
  images, labels = batch["image"].to('cpu'), batch["label"].to('cpu')
  print(images.shape, labels.shape, labels.unique())
  break
#%% check monai's sliding_window_inference
from monai.inferers import sliding_window_inference
roi_size = (160, 160, 160)
sw_batch_size = 4
outputs = sliding_window_inference(
    images, roi_size, sw_batch_size, predictor=unet_wmh.to('cpu').forward)
print(outputs.shape)

#%%
loss = loss_function(outputs, labels) 
#%% check what's wrong with monai's one-hot-encoded version of the `label` variable
# check the numpy version first
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)
outputs_onehot = post_pred(outputs)
labels_onehot = post_label(labels)
print(outputs_onehot.shape, labels_onehot.shape)

#%% check monai's dice evaluation
from monai.metrics import compute_meandice
compute_meandice(outputs_onehot,labels_onehot)

#%% visualize patch
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for i,i_d in enumerate(data_wmh.train_data):
  for j, j_d in enumerate(i_d):
    v.vol_peek(j_d['image'][0,:],overlay_vol=j_d['label'][0,:])
    break
  break
#%% check train dataloader
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for batch in data_wmh.train_dataloader():
  print(batch['image'].shape, batch['label'].shape)
  break
#%% check val dataloader
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for batch in data_wmh.val_dataloader():
  print(batch['image'].shape, batch['label'].shape)
  break
#%% check model inference
if not hasattr(data_wmh,'val_data'): data_wmh.setup()
for batch in data_wmh.val_dataloader():
  inference = unet_wmh.model(batch['image'].to(unet_wmh.device))
  batch['image'].shape, inference.shape
  break
#%% Test loss
loss = DiceLoss(include_background=True,to_onehot_y=True, softmax=True)
loss(inference, batch['label'].to(unet_wmh.device))
#%% Test loss/acc/dice metrics with model
import torchmetrics, torch
from monai.losses import DiceLoss
loss = DiceLoss(include_background=True,to_onehot_y=True, softmax=True)
acc = torchmetrics.Accuracy()
net = m.UNetWMH()
device = 'cuda'
for batch in data_wmh.val_dataloader():
  x, y = batch["image"].to(device), batch["label"].to(device)
  y_hat = net.forward(x)
  print(x.type(), x.shape, y.type(), y.shape, y_hat.type(), y_hat.shape)
  loss(y_hat, y)
  pred = y_hat.argmax(axis=1,keepdim=True).type(torch.long).to('cpu')
  gt = y.type(torch.long).to('cpu')
  acc(pred, gt)
  break
#%% forward and model result are the same
y_hat = net.forward(x)
y_hat_2 = net.model(x)
(y_hat - y_hat_2).norm()
#%%
v.vol_peek(x[0,:].squeeze().cpu(),overlay_vol=pred[0].squeeze())
#%%
v.vol_peek(x[0,:].squeeze().cpu(), overlay_vol=gt[0].squeeze())
#%% My own implementation of dice (y_hat: one_hot_encoded), y: normal
from med_deeplearning.models.model_evaluation import get_dice_label
label_no = y_hat.shape[1]
label_list = list(range(1,label_no))
get_dice_label(y_hat, y, labels=label_list)
#%% Test dice score from lightning (doesn't yet make sense either) (y_hat shall be one-hot-encoded)
# > Lightning reference: https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/functional/classification/dice.py#L63-L116
from torchmetrics.functional.classification.dice import dice_score
y_hat.shape,gt.shape
dice_score(y_hat,gt.cuda(),bg=False)
#%% test dice score from monai (include_background=True doesn't make sense)
# > monai reference: https://docs.monai.io/en/latest/metrics.html
# (pred_one_hot must be one-hot format, 1st dim is batch, example shape: [16, 3, 32, 32]. Values should be binarized.)
from monai.metrics import DiceMetric
from torch.nn import functional as F
dice_metric = DiceMetric(include_background=False)
pred_onehot = F.one_hot(y_hat.argmax(axis=1)).permute(0,4,1,2,3).cuda()
y_onehot = F.one_hot(y.squeeze().long()).permute(0,4,1,2,3).cuda()
dice_metric(pred_onehot, y_onehot)
#%% (non-onehot-encoded, does't make sense)
from monai.metrics import compute_meandice
compute_meandice(pred.cuda(), y.cuda(), include_background=True)
#%% test training process
# import datasets
# from importlib import reload
# reload(datasets)
# data_wmh = datasets.DataModuleWMH(image_paths_t1, label_paths_t2, label_transformations = [LabelValueScaled(keys=['label'],scale=1/1000)], batch_size=20)

if not hasattr(data_wmh,'train_data'): data_wmh.setup()

net = m.UNetWMH()
device = 'cuda'
for batch in data_wmh.train_dataloader():
  x, y = batch["image"].to(device), batch["label"].to(device)
  # y_hat = net.forward(x)
  print(x.type(), x.shape, y.type(), y.shape)

#%% visualize inference
for batch in data_wmh.val_dataloader():
  inference = unet_wmh.model(batch['image'].to(unet_wmh.device))
  # v.vol_peek(batch['image'][0,:].squeeze(),overlay_vol=batch['label'][0,:].squeeze())
  v.vol_peek(batch['image'][0,:].squeeze(),overlay_vol=inference[0].argmax(axis=0).cpu())
  break
#%% visualize 
for b in range(inference.shape[0]):
  vol = batch['image'][b,:].squeeze().detach()
  overlay_vol = inference[b,:].argmax(axis=0).squeeze().detach()
  print(vol.shape, overlay_vol.shape)
  v.vol_peek(vol, overlay_vol=overlay_vol)
  break

#%% check data_dict
data_dicts = [
    {"image": str(image_name), "label": str(label_name)}
    for image_name, label_name in zip(image_paths, label_paths)
]
train_files, val_files = data_dicts[:-2], data_dicts[-2:]
# train_files, val_files = random_split(data_dicts,[15,2])

#%% check RandCropByPosNegLabeld
import monai.transforms as t
pixdim=(1.0, 1.0, 1.2)
patch_size=(96, 96, 96)
axcodes="RAS"
trans = t.Compose([
  t.LoadImaged(keys=["image","label"]),
  t.AddChanneld(keys=['image','label']),
  t.Spacingd(keys=['image','label'],pixdim=pixdim, mode=("bilinear", "nearest"),),
  t.Orientationd(keys=["image", "label"], axcodes=axcodes),
  t.RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=patch_size,
    pos=1,
    neg=1,
    num_samples=8,
    image_key="image",
    image_threshold=0,)])
trans()
#%%
batch = data_wmh.train_files
out = trans(batch)
#%%
for i,sample in enumerate(out):
  for j, patch in enumerate(sample):
    print(i,j, patch['image'].shape, patch['label'].shape)
#%% check training dataloader
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for i, batch in enumerate(data_wmh.train_dataloader()):
    print(i, batch['image'].shape)

#%% test transform
train_transforms, val_transforms = datasets.transforms(label_transformations=[LabelValueScaled(keys=['label'],scale=1/1000)])
#%%
train_sample = train_transforms(data_wmh.train_files[0])[0]
train_sample['image'].shape, train_sample['label'].shape
#%%
for i in range(len(val_files)):
  val_sample_batch = val_transforms(data_wmh.val_files[i])
  for j in range(len(val_sample_batch)):
    val_sample = val_sample_batch[j]
    val_sample['image'].shape, val_sample['label'].shape
#%% test CacheDataset
train_data = CacheDataset(data=data_wmh.train_files, transform=trans, num_workers=15, cache_rate=1.)
#%%
train_data[0][0]['image'].shape
#%% test Dataloader
dl = DataLoader(train_data, batch_size=3, shuffle=True, num_workers=0)
for i, batch in enumerate(dl):
    print(i, batch['image'].shape, batch['label'].shape)
#%% test actual dataloader
if not hasattr(data_wmh,'train_data'): data_wmh.setup()
for i, batch in enumerate(data_wmh.train_dataloader()):
    print(i, batch['image'].shape, batch['label'].shape)

#%% train_dataloader
from monai.data import SmartCacheDataset, CacheDataset
from monai.data.utils import list_data_collate
from torch.utils.data import random_split, DataLoader
val_idx=-2
root_dir = f"/tmp/dma73/Data"
persistent_cache_dir = Path(root_dir,'persistent_cache')
persistent_cache_dir.mkdir(parents=True, exist_ok=True)
data_dicts = [{"image": str(image_name), "label": str(label_name)}
    for image_name, label_name in zip(image_paths, label_paths)]
train_files, val_files = data_dicts[:val_idx], data_dicts[val_idx:]
# train_data = PersistentDataset(data=[data_dicts[i] for i in list(range(len(data_dicts)))], transform=val_transforms, cache_dir=persistent_cache_dir)
# train_data = SmartCacheDataset(data=data_dicts, transform=val_transforms, cache_rate=0.5, replace_rate=0.5)
train_data = CacheDataset(data=[data_dicts[i] for i in list(range(len(data_dicts)))], transform=val_transforms, cache_rate=1.0, num_workers=4)
train_dl = DataLoader(train_data, batch_size=2, shuffle=False, collate_fn=list_data_collate, num_workers=0)

for b, batch in enumerate(train_dl):
  print(b, batch['image'].shape, batch['label'].shape)
  # break

#%% turns out data 15/16 have problems for the RandCropByPosNegLabeld
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
pixdim=(1.0, 1.0, 1.2)
patch_size=(96, 96, 96)
axcodes="RAS"
label_transformations=[]
data=[data_dicts[i] for i in list(range(len(data_dicts)))]
# val_transforms = t.Compose([
#   t.LoadImaged(keys=["image","label"]),
#   t.AddChanneld(keys=['image','label']),
#   t.Spacingd(keys=['image','label'],pixdim=pixdim, mode=("bilinear", "nearest"),),
#   t.Orientationd(keys=["image", "label"], axcodes=axcodes),
#   t.RandCropByPosNegLabeld(
#       keys=["image", "label"],
#       label_key="label",
#       spatial_size=patch_size,
#       pos=1,
#       neg=1,
#       num_samples=8,
#       image_key="image",
#       image_threshold=0),
#   t.ToTensord(keys=["image","label"]),
#   ])
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=3,
            image_key="image",
            image_threshold=0,
        ),
        # ScaleIntensityRanged(
        #     keys=["image"], a_min=-57, a_max=164,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        # ToTensord(keys=["image", "label"]),
    ])
out = val_transforms(data)

#%% 
import numpy as np
trans = t.Compose([LabelValueScaled(keys=["label"],scale=1/1000),
                  t.CastToTyped(keys=["label"],dtype=np.int32),
                  t.ToTensord(keys=["image","label"]),
                  ])
for i, b in enumerate(out):
  for j, s in enumerate(b):
    print(i, j, s['image'].shape, s['label'].shape, np.unique(s['label']), s['label'].dtype)
    o = trans(s)
    print(i, j, o['image'].shape, o['label'].shape, o['label'].unique(), o['label'].type())   
#%%
trans = t.Compose([t.CastToTyped(keys=["label"],dtype=np.uint32)])
for i, b in enumerate(out):
  for j, s in enumerate(b):
    o = trans(s)
    v.vol_peek(o['image'].squeeze(), overlay_vol=o['label'].squeeze())
#%%
trans = t.Compose([LabelValueScaled(keys=["label"],scale=1/1000),
                  t.CastToTyped(keys=["label"],dtype=np.int32),
                  #t.ToTensord(keys=["image","label"]),
                  ])
trans(out[0])
#%%
val_files
#%% debug val_dataloader
from monai.data.utils import list_data_collate
from torch.utils.data import random_split, DataLoader
val_idx=-2
root_dir = f"/tmp/dma73/Data"
persistent_cache_dir = Path(root_dir,'persistent_cache')
persistent_cache_dir.mkdir(parents=True, exist_ok=True)
data_dicts = [{"image": str(image_name), "label": str(label_name)}
    for image_name, label_name in zip(image_paths, label_paths)]
train_files, val_files = data_dicts[:val_idx], data_dicts[val_idx:]
valid_data = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache_dir)
val_dl = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=list_data_collate, num_workers=1)

for batch in val_dl:
  print(batch['image'].shape, batch['label'].shape)
  break

#%% visualize image/label pairs
v.vol_peek(vol=data_dicts[0]['image'], overlay_vol=data_dicts[0]['label'])

#%% test transformations
T = t.LoadImaged(keys=["image","label"])
output = T(data_dicts[0])
T = t.AddChanneld(keys=['image','label'])
output = T(output)
print(output['image'].shape)
T = LabelValueScaled(keys=['label'],scale=1/1000) # 1/1000 for T2 Hyper-Intensity, 1 for T1 Hypo-intensity
#%% setup transforms for training and validation
# def transforms():
#     train_transforms = t.Compose([
#         # deterministic
#         t.LoadImaged(keys=["image","label"]),
#         t.AddChanneld(keys=['image','label']),
#         t.Spacingd(keys=['image','label'],pixdim=(1.0, 1.0, 1.2), mode=("bilinear", "nearest"),),
#         t.Orientationd(keys=["image", "label"], axcodes="RAS"),
#         # scale label value
#         LabelValueScaled(keys=['label'],scale=1),
#         # randomly crop out patch samples from big image based on pos / neg ratio
#         # the image centers of negative samples must be in valid image area
#         t.RandCropByPosNegLabeld(
#             keys=["image", "label"],
#             label_key="label",
#             spatial_size=(96, 96, 96),
#             pos=1,
#             neg=1,
#             num_samples=8,
#             image_key="image",
#             image_threshold=0,
#         ),
#         t.ToTensord(keys=["image","label"]),
#     ])
#     val_transforms = t.Compose([
#         # deterministic
#         t.LoadImaged(keys=["image","label"]),
#         t.AddChanneld(keys=['image','label']),
#         t.Spacingd(keys=['image','label'],pixdim=(1.0, 1.0, 1.2), mode=("bilinear", "nearest"),),
#         t.Orientationd(keys=["image", "label"], axcodes="RAS"),
#         # scale label value
#         LabelValueScaled(keys=['label'],scale=1),
#         # randomly crop out patch samples from big image based on pos / neg ratio
#         # the image centers of negative samples must be in valid image area
#         t.ToTensord(keys=["image","label"]),
#     ])
#     return train_transforms, val_transforms

#%% Use PersistentDataset
# > ref: /home/dma73/Code/medical_image_analysis/cohorts/monai/tutorials/acceleration/dataset_type_performance.ipynb
# ?PersistentDataset
# root_dir = f"/scratch/dma73/Data"
root_dir = f"/tmp/dma73/Data"
persistent_cache_dir = Path(root_dir,'persistent_cache')
persistent_cache_dir.mkdir(parents=True, exist_ok=True)

#%% enable deterministic training
monai.utils.set_determinism(seed=0)
# %% normal dataset
import dataset
from importlib import reload
reload(dataset)
train_trans, val_trans = dataset.transforms()
train_ds = Dataset(data=train_files, transform=train_trans)
val_ds   = Dataset(data=val_files, transform=val_trans)
#%% persistant dataset
train_trans, val_trans = dataset.transforms()
train_persistence_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache_dir)
val_persitence_ds   = PersistentDataset(data=val_files, transform=val_trans, cache_dir=persistent_cache_dir)

# %%
(
    persistence_epoch_num,
    persistence_total_time,
    persistence_epoch_loss_values,
    persistence_metric_values,
    persistence_epoch_times,
) = train_process(train_persistence_ds, val_persitence_ds)
print(
    f"total training time of {persistence_epoch_num}"
    f" epochs with persistent storage Dataset: {persistence_total_time:.4f}"
)
# %%
