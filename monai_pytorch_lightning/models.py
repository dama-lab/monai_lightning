#%% import libraries
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric # (depreciated): compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai import transforms as t
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

#%% Refs:
'''
# lightning: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb

# Logging: https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-from-a-lightningmodule
# === The log() method has a few options: ===
# - on_step: Logs the metric at the current step. Defaults to True in training_step(), and training_step_end().
# - on_epoch: Automatically accumulates and logs at the end of the epoch. Defaults to True anywhere in validation or test loops, and in training_epoch_end().
# - prog_bar: Logs to the progress bar.
# - logger: Logs to the logger like Tensorboard, or any other custom logger passed to the Trainer.
'''

#%%
class Model(pl.LightningModule):
  def __init__(self):
    # define model here
    return

  def forward(self, x):
    return

  def configure_optimizers(self):
    return

  def training_step(self, train_batch, batch_idx):
    return

  def validation_step(self, valid_batch, batch_idx):
    return

#%% calculate the dice score
def get_dice_label(input:torch.Tensor, targs:torch.Tensor, labels:(list,int), iou:bool=False, eps:float=1e-8) -> torch.Tensor:
    '''
    Dice coefficient metric for multiple layers (classic for segmentation problems). If iou=True, return Intersection over Union (IoU) metric,
    Ref: https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53

    > dim[0]: sample number
    > dim[1]: flattened pixels
    
    #%% Customized metrics: dice_list
    # step 1 build metrics only using the dice for specific layers
    # Type checking: https://stackoverflow.com/questions/19684434/best-way-to-check-function-arguments-in-python/37961120
    # Collection/Union Ref: https://github.com/fastai/fastai/blob/master/fastai/core.py
    '''
    n = targs.shape[0]

    # To prevent error message of RuntimeError: “argmax_cuda” not implemented for ‘Bool’
    # Ref: https://discuss.pytorch.org/t/runtimeerror-argmax-cuda-not-implemented-for-bool/58808/4

    # input = input.long() # equal to: .type(torch.LongTensor)
    # targs = targs.to('cpu')

    dice = 0

    # convert label to list in case of int (Check type class of label Ref: https://stackoverflow.com/questions/152580/whats-the-canonical-way-to-check-for-type-in-python)
    if isinstance(labels, int):
        labels = [labels]

    for label in labels:
        # argmax along the channel dimension, -1 means PyTorch to determine automatically
        ### to-be-deleted # input = (input.argmax(dim=1) == label).type(torch.LongTensor).view(n, -1)

        input = (input.argmax(dim=1) == label).long().view(n, -1)
        targs = (targs == label).long().view(n,-1)

        intersect = (input * targs).sum(dim=1).float()
        union = (input + targs).sum(dim=1).float()

        if not iou: # dice
            l = 2. * intersect / union
        else: # what's this?
            l = intersect / (union-intersect+eps)

        l[union == 0.] = 1 # If current label non-exist in any sample of the current batch, assign dice to 1
        dice += l.mean()

    if len(labels) > 0:
        dice = dice / len(labels)
    # else: dice = 0
    return dice

#%% calculate accuracy
def accuracy(pred, gt, argmax_pred=True):
  '''
  # Calculate tp/fp/tn/fn
  > /project/6026508/dma73/Tools/miniconda3/envs/fastai/lib/python3.8/site-packages/torchmetrics/functional/classification/accuracy.py
    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)
  
  # calculate numerator/denominator
  > /project/6026508/dma73/Tools/miniconda3/envs/fastai/lib/python3.8/site-packages/torchmetrics/functional/classification/accuracy.py
    numerator = tp + tn
    denominator = tp + tn + fp + fn

  # calculate accuracy score
  > /project/6026508/dma73/Tools/miniconda3/envs/fastai/lib/python3.8/site-packages/torchmetrics/classification/stat_scores.py
    weights=None if average != "weighted" else tp + fn,
    scores = weights * (numerator / denominator)
  '''
  acc = torchmetrics.Accuracy()
  device = torch.device('cpu')
  datatype = torch.long
  if argmax_pred is True: 
    pred = pred.argmax(axis=1,keepdim=True)
  pred = pred.type(datatype).to(device)
  gt   = pred.type(datatype).to(device)
  return acc(pred, gt)

#%%
def cal_mean_dice(y_hat, y, include_bg=False):
  '''
  y_hat: one_hot_encoded (e.g. [12,2,96,96])
  y =    argmaxed        (e.g. [12,1,96,96])
  '''
  CLASS_CHANNEL_DIM = 1
  label_no = y_hat.shape[CLASS_CHANNEL_DIM]
  # if include_bg is True  (1): start from 0st channel
  # if include_bg is False (0): start from 1st channel
  from_channel = int(not include_bg)
  label_list = list(range(from_channel, label_no))
  return get_dice_label(y_hat, y, labels=label_list)

def unet_monai(in_channels, out_channels, dimensions=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, kernel_size=3):
  '''return a monai's UNet with some default parameters
  - for 2D: dimensions=2
  '''
  return UNet(dimensions=dimensions,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        num_res_units=num_res_units,
        norm=Norm.BATCH)

#%% Model_WMH
class UNet3DModule(pl.LightningModule):
  ''' Ref: spleen_segmentation_3d_lightning
  https://colab.research.google.com/github/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb#scrollTo=GYyFYlkbntYB
  https://colab.research.google.com/drive/1swTt4hH9gguJ1ovwfwcpC5q9P0vkBJb7#scrollTo=vVI2lj36wp4Z
  '''
  def __init__(self, net, lr=1e-4, 
               loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True),
               roi_size = (64, 64, 64),
               optimizer_class=torch.optim.AdamW,
               automatic_optimization = True,
               sw_batch_size=2,
               device=torch.device("cuda:0"),
               val_device='cpu', verbose=True):
    '''
    - roi_size: sample value
      3D brain: (64, 64, 64)
      2D retina: (512,512,1)
    - automatic_optimization:
      two options: 1. automatic; 2. manual
    # unet = UNet(
    #     dimensions=3,
    #     in_channels=1,
    #     out_channels=2,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_ units=2,
    #     norm=Norm.BATCH,
    # ).to(device)
    
    '''
    super().__init__()
    self.sw_device = device
    self.val_device = val_device
    self._model = net.to(device)
    self.optimizer_class = optimizer_class
    self.lr = lr
    self.loss_function = loss_function # include_background=True
    self.roi_size = roi_size
    self.sw_batch_size = sw_batch_size
    self.post_pred = t.Compose([t.EnsureType(), t.AsDiscrete(argmax=True, to_onehot=True, n_classes=net.out_channels)])
    self.post_label = t.Compose([t.EnsureType(), t.AsDiscrete(to_onehot=True, n_classes=net.out_channels)])
    self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False) # `get_not_nans=True` will return a second value
    self.best_val_dice = 0
    self.best_val_epoch = 0
    self.verbose = verbose
    # for learning rate schedular
    # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling
    self.automatic_optimization = automatic_optimization
    
  def forward(self, x):
    return self._model(x)

  def forward_val(self, x):
    '''
    - for 3D: the same as 2D
    - for 2D: need to customize the design
    '''
    return self._model(x)

  def configure_optimizers(self):
    '''Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you’d need one. But in the case of GANs or similar you might have multiple.
    Any of these 6 options. (I'm currently using the "Two lists" option)
    - Single optimizer.
    - List or Tuple of optimizers.
    - Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).
    - Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.
    - Tuple of dictionaries as described above, with an optional "frequency" key.
    - None - Fit will run without any optimizer.
    Ref: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers'''
    # optimizer
    optimizer = self.optimizer_class(self.parameters(),lr=self.lr)
  
    # lr_schesuler
    schedular_reduceonplateau = {
      "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
      "monitor":   'val_loss',
      "frequency": 1,
      } # frequency: how often n epochs

    optimizers = [optimizer]
    lr_schedulers = [ schedular_reduceonplateau ]
    return optimizers, lr_schedulers
    # return torch.optim.SGD(self.parameters(),lr=self.lr)

  def training_step(self, train_batch, batch_idx):
    images, labels = train_batch["image"], train_batch["label"] #.to(device)
    output = self.forward(images)
    loss = self.loss_function(output, labels)
    # logs metrics for each training_step,
    # and average across the epoch, to the progress bar and logger
    self.log('train_loss', loss.item(), prog_bar=True, logger=True) # , on_step=True, on_epoch=True

    # train_acc = accuracy(output, labels)
    # self.log('train_acc', train_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    
    return {'loss': loss} # , 'train_accuracy': train_acc

  def validation_step(self, batch, batch_idx):
      '''implementation from monai
      Ref: /home/dma73/Code/medical_image_analysis/cohorts/Learning/monai/MONAI/tutorials/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
      '''
      images = batch["image"].to(self.val_device)
      labels = batch["label"].to(self.val_device)
      
      with torch.no_grad():
        outputs = sliding_window_inference(images, self.roi_size, self.sw_batch_size, predictor=self.forward_val, sw_device=self.sw_device,  device=self.val_device).to(self.val_device)
        # calculate losses
        loss = self.loss_function(outputs, labels)
        # calculate metrics
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        valid_dice = self.dice_metric(y_pred=outputs, y=labels)
        # seems there were nan in the validation dices
        valid_dice = valid_dice[valid_dice.isnan().logical_not()].mean()
      # will only log mean loss/dice at "validation_epoch_end"
      # self.outputs = outputs # for debug only
      return {"val_loss": loss, "val_number": len(outputs), "val_dice": valid_dice}

  def training_epoch_end(self, outputs):
    ''' Learning rate scheduling for reduceonplatue [manual]
    (probably duplicated with: `configure_optimizers`)
    Ref: https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling-manual
    '''
    sch = self.lr_schedulers()
    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
      sch.step(self.trainer.callback_metrics['val_loss'])



  def validation_epoch_end(self, outputs):
    ''' postprocessing validation step output (MONAI's implementation)
    '''
    val_loss, num_items = 0, 0
    for output in outputs:
      # val_dice += output["val_dice"].sum().item()
      val_loss += output["val_loss"].sum().item()
      num_items += output["val_dice"].numel()
    mean_val_dice = self.dice_metric.aggregate().item()
    self.dice_metric.reset() # reset dice metrics after calculating the mean
    mean_val_loss = torch.tensor(val_loss / num_items)
   
    if mean_val_dice > self.best_val_dice:
      self.best_val_dice = mean_val_dice
      self.best_val_epoch = self.current_epoch

    if self.verbose is True:
      output_str = f"(best/current) epoch {self.best_val_epoch}/{self.current_epoch} - "
      output_str += f"mean dice: {mean_val_dice:.4f}/{self.best_val_dice:.4f}"
      print(output_str)
      
    tensorboard_logs = {"val_dice": mean_val_dice, "val_loss": mean_val_loss}
    self.log('val_loss', mean_val_loss, prog_bar=True, logger=True)
    self.log('val_dice', mean_val_dice, prog_bar=True, logger=True)
    return {"log": tensorboard_logs}
  
  #%% =============================================
  #%% use self-implemented validation calculation
  #%% =============================================
  
  def validation_step_self(self, valid_batch, batch_idx):
    '''
    my own implementation of validation step
    look below for monai's implementation
    '''
    x, y = valid_batch["image"], valid_batch["label"] #.to(device)
    y_hat = self.forward(x)
    loss = self.loss_function(y_hat, y)
    # logs metrics for each valid_epoch,
    self.log('val_loss', loss, prog_bar=True, logger=True) # , on_step=True, on_epoch=True

    # validation dice
    valid_dice = cal_mean_dice(y_hat, y)
    self.log('valid_dice', valid_dice, prog_bar=True, logger=True)
    # valid_acc = accuracy(y_hat, y) 
    # self.log('valid_acc', valid_acc, prog_bar=True, logger=True)
    return {'val_loss': loss, "val_dice": valid_dice} # , 'valid_accuracy': valid_acc 
  
  def validation_epoch_end_self(self, outputs):
    ''' postprocessing validation step output (My own implementation)
    '''

    val_dice, val_loss, num_items = 0, 0, 0
    for output in outputs:
      val_dice += output["val_dice"].sum().item()
      val_loss += output["val_loss"].sum().item()
      num_items += output["val_dice"].numel()
    mean_val_loss = torch.tensor(val_loss / num_items)
    mean_val_dice = torch.tensor(val_dice / num_items)

    # update best mean validation dice
    if mean_val_dice > self.best_val_dice:
      self.best_val_dice = mean_val_dice
      self.best_val_epoch = self.current_epoch

    # print
    if self.verbose is True:
      output_str = f"(best/current) epoch {self.best_val_epoch}/{self.current_epoch} - "
      output_str += f"mean dice: {mean_val_dice:.4f}/{self.best_val_dice:.4f}"
      print(output_str)
      
    # log
    logs = {"val_dice": mean_val_dice, "val_loss": mean_val_loss}
    # use the newly introduced log function
    self.log('val_loss', mean_val_loss, prog_bar=True, logger=True)
    self.log('val_dice', mean_val_dice, prog_bar=True, logger=True)
    return {"log": logs}

# %%
class UNet2DModule(UNet3DModule):
  def __init__(self, net, lr=1e-4, 
               loss_function=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True),
               roi_size = (512, 512, 1),
               optimizer_class=torch.optim.AdamW, 
               device=torch.device("cuda:0"),
               val_device='cpu',
               sw_batch_size=2,
               verbose=True):
    ''' 
    - val_device: device to store the whole validation data to be feed into the sliding_window_inference on "device" (default is 'cpu')
    '''
    super().__init__(net=net, lr=lr, loss_function=loss_function, roi_size=roi_size, optimizer_class=optimizer_class, sw_batch_size=sw_batch_size, device=device, val_device=val_device, verbose=verbose)

  def forward_val(self, x):
    '''
    squeeze last dimension to 2D before feeding into the model
    unsqueeze last dimension to 3D after feeding into the model
    '''
    return self._model(x.squeeze(-1)).unsqueeze(-1)

  # #%% Ref: /home/dma73/Code/medical_image_analysis/cohorts/Learning/monai/MONAI/tutorials/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
  # def validation_step(self, batch, batch_idx):
  #     '''implementation from monai'''
  #     images = batch["image"].to(self.val_device)
  #     labels = batch["label"].to(self.val_device)
  #     # ensure everything is in eval() mode (i.e. no tensor grad calculation)
  #     # https://github.com/Project-MONAI/MONAI/issues/1189
  #     with torch.no_grad():
  #       outputs = sliding_window_inference(images, self.roi_size, self.sw_batch_size, predictor=self.forward_val, sw_device=self.sw_device,  device=self.val_device).to(self.val_device)
  #       # calculate losses
  #       loss = self.loss_function(outputs, labels)
  #       # calculate dice metrics
  #       outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
  #       labels = [self.post_label(i) for i in decollate_batch(labels)]
  #       valid_dice = self.dice_metric(y_pred=outputs, y=labels)
  #       # seems there were nan in the validation dices
  #       valid_dice = valid_dice[valid_dice.isnan().logical_not()].mean()
  #       # will only log mean loss/dice at "validation_epoch_end"
  #       # self.outputs = outputs # for debug only
  #     return {"val_loss": loss, "val_number": len(outputs), "val_dice": valid_dice}