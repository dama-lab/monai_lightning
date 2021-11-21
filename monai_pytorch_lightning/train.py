#%%
import glob
import os
import argparse
import pathlib
import shutil
import tempfile
import time

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


#%% %%%%%%%% Lightning %%%%%%%%

# %% pytorch_lightning callbacks
class PrintingCallbacks(Callback):

  def on_init_start(self, trainer):
    print('Starting to init trainer!')

  def on_init_end(self, trainer):
    print('trainer is init now')

  def on_train_start(self, trainer, pl_module):
    print(f"Training is started!, learning rate = {pl_module.lr}")
    
  def on_train_end(self, trainer, pl_module):
    print("Training is done.")

#%% train from lightning
def lightning_trainer(default_root_dir,log_dir=None, gpus=[0], auto_lr_find=True, accelerator='dp',max_epochs=600, log_every_n_steps=50, flush_logs_every_n_steps=100, csv_log_name="csv_log", tb_log_name="tensorboard", monitor='val_loss', mode="min", save_top_k=1, exp_name="", verbose=True):
  '''
  model: lightning module
  data:  lightning data module
  mode: "min" for "val_loss", "max" for "val_dice"
  '''

  #%% define logger
  if log_dir is None: log_dir = f"{default_root_dir}/logs"
  tb_logger  = pl.loggers.TensorBoardLogger(save_dir=f'{log_dir}/{tb_log_name}')
  csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, name=csv_log_name)

  #%% define callbacks
  # > ref: https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference
  # > ref: https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html
  checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode, verbose=verbose, dirpath=f"{default_root_dir}/checkpoints", save_top_k = save_top_k, filename = f"{exp_name}-"+"{val_dice:.4f}_{epoch}") # save top n model # {val_loss:.4f}_

  early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=200)
  callbacks=[checkpoint_callback, PrintingCallbacks(),early_stopping_callback]
  # saved best checkpoint can be retrieved through: `checkpoint_callback.best_model_path`

  #%% define/initialize trainier
  trainer = pl.Trainer(default_root_dir=default_root_dir,
                      gpus=gpus, 
                      accelerator=accelerator, 
                      max_epochs=max_epochs, 
                      auto_lr_find=auto_lr_find, 
                      log_every_n_steps=log_every_n_steps,
                      flush_logs_every_n_steps=flush_logs_every_n_steps, 
                      callbacks=callbacks, 
                      checkpoint_callback=True,
                      num_sanity_val_steps=2,
                      logger=[csv_logger, tb_logger])

  return trainer

#%% train_lightning
def setup_options(input_args=None):
  '''Setup input argument options for training'''
  # Training settings
  parser = argparse.ArgumentParser(description='Segmentation')
  ## pytorch-lightning setings
  parser.add_argument('--num_gpus', type=int, default=-1, help="set to -1 for all available gpus")
  parser.add_argument('--log_dir', type=str, default=None, help="log directory for run")

#%%
# def train_lightning(input_args=None):



# %% inference



#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% %%%%%%%% MONAI + Pure PyTorch %%%%%%%%
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% sample training process created by monai
def train_monai(train_ds, val_ds):
    
    from monai.data import list_data_collate
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceLoss
    from monai.metrics import compute_meandice
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet
    from monai.transforms import AsDiscrete

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    #%%
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

    epoch_num = 100
    val_interval = 1  # do validation for every epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    epoch_times = list()
    total_start = time.time()
    for epoch in range(epoch_num):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
            #     f" step time: {(time.time() - step_start):.4f}"
            # )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    with torch.no_grad():
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model
                        )
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
        print(f"time of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        f" total time: {(time.time() - total_start):.4f}"
    )
    return (
        epoch_num,
        time.time() - total_start,
        epoch_loss_values,
        metric_values,
        epoch_times,
    )