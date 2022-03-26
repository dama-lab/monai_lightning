#%%
# import glob
import os
import argparse
from pathlib import Path
import time
# import shutil
# import tempfile

import torch
import monai
import monai.transforms as t
from monai.data import DataLoader, Dataset, nifti_writer, write_nifti
from monai.inferers import sliding_window_inference

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# import other core library components
import transforms, datasets, models, train, visualization as viz
from HelperFunctions.data_proc import oct_proc

# %% To-do
# - layer-wise differential learning rate:
#   https://docs.monai.io/en/latest/optimizers.html

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
def lightning_trainer(default_root_dir,log_dir=None, gpus=[0], auto_lr_find='lr', accelerator='dp',max_epochs=600, log_every_n_steps=50, precision=16, flush_logs_every_n_steps=100, csv_log_name="csv_log", tb_log_name="tensorboard", monitor='val_loss', mode="min", save_top_k=1, exp_name="", patience=50, verbose=True):
  '''
  model: lightning module
  data:  lightning data module
  mode: "min" for "val_loss", "max" for "val_dice"
  precision: AMP 16 precision training # https://www.youtube.com/watch?v=fq7gAacJirQ
  '''

  #%% define logger
  if log_dir is None: log_dir = f"{default_root_dir}/logs"
  tb_logger  = pl.loggers.TensorBoardLogger(save_dir=f'{log_dir}/{tb_log_name}')
  csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, name=csv_log_name)

  #%% define callbacks
  # > ref: https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference
  # > ref: https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html
  checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode, verbose=verbose, dirpath=f"{default_root_dir}/checkpoints", save_top_k = save_top_k, filename = f"{exp_name}-"+"{val_dice:.4f}_{epoch}") # save top n model # {val_loss:.4f}_

  early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=patience)
  callbacks=[checkpoint_callback, PrintingCallbacks(),early_stopping_callback]
  # saved best checkpoint can be retrieved through: `checkpoint_callback.best_model_path`

  #%% define/initialize trainier
  trainer = pl.Trainer(default_root_dir=default_root_dir,
                      gpus=gpus, precision=precision,
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

class segmentation_pipeline():
  def __init__(self, default_root_dir,
              dimensions = 2,
              patch_size = (480,480,1),
              sw_batch_size = 256,
              device = "cuda",
              val_device = 'cpu',
              ):
    '''The main segmentation engine to perform the segmentation training/inference pipeline
    - mode = train/inference
    - image_paths, label_paths = get_image_label_paths()
    '''
    self.dimensions = dimensions
    self.patch_size = patch_size
    self.default_root_dir = default_root_dir
    self.sw_batch_size = sw_batch_size
    self.device = device
    self.val_device = val_device

    if self.dimensions == 2:
      self.module = models.UNet2DModule
    elif self.dimensions == 3:
      self.module = models.UNet3DModule
    # end of __init__ function
    
  # %% common method steps for both training / inference
  # %% create model
  def create_model(self, out_channels, in_channels=1, num_res_units=2, lr=1e-3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), optimizer_class=monai.optimizers.Novograd):
    '''create: self.net / self.model'''
    # define network architecture
    self.net = models.unet_monai(dimensions=self.dimensions, in_channels=in_channels, out_channels=out_channels, num_res_units=num_res_units, channels=channels, strides=strides)
    # define model
    self.model = self.module(net=self.net, lr=lr, optimizer_class=optimizer_class, roi_size=self.patch_size, sw_batch_size=self.sw_batch_size, device=self.device, val_device=self.val_device)

  # %% create data module
  def create_data(self, image_paths, label_paths=None, batch_size_train=32, batch_size_val=1, batch_size_test=1, dataset_type='CacheDataset', num_workers=0, num_workers_cache=0, pixdim=(1.0, 1.0, 1.2), axcodes="RAS", label_transformations=[], crop_forground_select_fn=None):
    ''' create data module'''
    # create transform
    if self.dimensions == 2:
      self.train_transforms, self.val_transforms, self.test_transforms = transforms.transforms_2d(patch_size=self.patch_size, crop_forground_select_fn=crop_forground_select_fn)
    elif self.dimensions == 3:
      self.train_transforms, self.val_transforms, self.test_transforms = transforms.transforms_3d(pixdim=pixdim, patch_size=self.patch_size, axcodes=axcodes, label_transformations=label_transformations)
    
    # if single path string is provided, convert image_paths to list
    if not isinstance(image_paths, list):
      image_paths = [image_paths]
    if not isinstance(label_paths, list):
      label_paths = [label_paths]

    # Create data
    self.data = datasets.DataModule(image_paths=image_paths, label_paths=label_paths, train_transforms=self.train_transforms, val_transforms=self.val_transforms, test_transforms=self.test_transforms, batch_size_train=batch_size_train, batch_size_val=batch_size_val, batch_size_test=batch_size_test, dataset_type=dataset_type, num_workers=num_workers, num_workers_cache=num_workers_cache)
    
    # if self.mode == 'train':  
    #   self.data.setup(val_idx=val_idx)
    # elif self.mode == 'inference':
    #   test_dataloader = self.data.test_dataloader()

  # %% training specific blocks (data_module, trainer)    
  def train(self, exp_name, val_idx=-1, max_epochs=1000, monitor='val_dice', mode='max', save_top_k=1, patience=100, find_lr=True, fit=True):
    # if self.mode != 'train':
    #   print("self.mode not in 'train' mode"); return
    # setup the data
    print("===== initializing the training/validation data ===== ....")
    self.data.setup(val_idx=val_idx)

    # create lightning trainer
    log_dir=f"{self.default_root_dir}/logs/{exp_name}"
    trainer = train.lightning_trainer(default_root_dir=self.default_root_dir, 
                                      log_dir=log_dir,
                                      auto_lr_find='lr',
                                      max_epochs=max_epochs, 
                                      exp_name=exp_name, 
                                      monitor=monitor,
                                      mode=mode,
                                      save_top_k=save_top_k,
                                      patience=patience)
    # find the best learning rate
    # Ref: https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html
    if find_lr == True:
      print("===== finding the best learning rate ....")
      print("lr before auto_lr_find: ", self.model.lr)
      lr_finder = trainer.tuner.lr_find(self.model, self.data)
      fig = lr_finder.plot(suggest=True) # plot
      fig.show()
      Path(log_dir).mkdir(exist_ok=True, parents=True)
      fig.savefig(f"{log_dir}/{exp_name}_lr={lr_finder.suggestion():.6f}_lr_finder.jpg")
      print("suggested learning rate: ", lr_finder.suggestion())
      self.model.hparams.learning_rate = lr_finder.suggestion()
      self.model.lr = lr_finder.suggestion()
      # alternatively:
      # trainer.tune(model, data)
      print("lr after auto_lr_find: ", self.model.lr)

    # %% Start training with learning rate = {lr}
    if fit == True:
      print(f'===== Start training with learning rate = {self.model.lr}...')
      trainer.fit(self.model, self.data)

  def inference(self, ckpt_path=None,
                seg_dir = None,
                qc_dir  = None,
                ext_len = -1,
                qc_seg  = True,
                qc_surf = False,
                strict  = False,
                linewidth=0.5,
                overwrite = False,
                ):
    '''inference
    - ext_len: -1=.nii, -2=.nii.gz
    - strict:  whether to load the weight key name in strict mode
    '''
    # load model checkpoints if specified
    # otherwise can preload the pre-trained model weight from checkpoint file through: `self.model.load_from_checkpoint(checkpoint_path=ckpt_path, net=segmentor.net, map_location=segmentor.device, strict=True, verbose=True)
    if ckpt_path is not None:
      self.model.load_from_checkpoint(checkpoint_path=ckpt_path, net=self.net, map_location=self.device, strict=strict, verbose=True);


    if seg_dir == None:
      seg_dir = f"{self.default_root_dir}/seg"
    if qc_dir == None:
      qc_dir = f"{self.default_root_dir}/quiqkcheck"

    Path(seg_dir).mkdir(exist_ok=True, parents=True)

    test_dataloader = self.data.test_dataloader()
    
    # %% start inference
    for i, test_batch in enumerate(test_dataloader):
      img = test_batch['image'] #.to(val_device) # 'cpu'
      img_path = test_batch['image_meta_dict']['filename_or_obj'][0]
      fname = os.path.basename(img_path)
      scan_name = '_'.join(fname.split('.')[:ext_len])
      seg_path = f"{seg_dir}/{scan_name}.nii.gz"

      print(f"{i}/{len(test_dataloader)}: {scan_name}")

      if os.path.isfile(seg_path):
        print(f"--- segmentation file exits: `{seg_path}`, , skipping ...")
        seg = seg_path
        continue
      # run Inference
      with torch.no_grad():
        # run sliding window inference
        print("--- running sliding_window_inference ...")
        test_outputs = sliding_window_inference(inputs=img, roi_size=self.patch_size, sw_batch_size=self.sw_batch_size, predictor=self.model.to('cuda').forward_val, overlap=0.1, sw_device="cuda",device="cpu")

        # convert segmentation (post conversion: transpose)
        seg = test_outputs.argmax(dim=1).squeeze().cpu()
        # seg = seg.cpu()
      
      # save auto-seg resutls
      if not os.path.isfile(seg_path) or overwrite is True:
        print("--- Saving segmentation ...")
        write_nifti(seg.permute([1,0,2]).numpy(), file_name=seg_path)

      # prepare outputs
      vol = img.squeeze().detach().cpu()

      # save seg quickcheck
      if qc_seg is True:
        qc_seg_dir = f"{qc_dir}/seg"
        Path(qc_seg_dir).mkdir(exist_ok=True, parents=True)

        qc_seg_path = f"{qc_seg_dir}/{scan_name}.png"
        if not os.path.isfile(qc_seg_path) or overwrite is True:
          print("--- Saving seg quickcheck ...")
          fig = viz.vol_peek(vol=vol, overlay_vol=seg, filepath=qc_seg_path, fig_title=scan_name, show=False, close_figure=True) 

      # save retina-specific qc:
      if qc_surf == True:
        qc_surf_dir = f"{qc_dir}/surf"
        Path(qc_surf_dir).mkdir(exist_ok=True, parents=True)

        qc_surf_path  = f"{qc_surf_dir}/{scan_name}.png"
        # save quickcheck surface results
        if not os.path.isfile(qc_surf_path) or overwrite is True:
          print("--- Saving layer surface quickcheck ...")
          surf = oct_proc.layer2surf(seg)
          fig = viz.vol_peek(vol, overlay_surf=surf, linewidth=linewidth, filepath=qc_surf_path, show=True, close_figure=True)
  
    # return the last segmentation output (for debugging)
    return vol, seg


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