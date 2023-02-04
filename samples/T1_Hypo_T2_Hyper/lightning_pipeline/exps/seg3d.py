#%% load libraries
print('  >>> importing libraries...')
import os, sys
from pathlib import Path
import argparse
from datetime import datetime
from glob import glob

import monai.transforms as t
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

# %% import customized libraries 
sys.path.append(os.path.abspath('../../../../monai_pytorch_lightning'))

import datasets, train, utils, models
from transforms import LabelValueRemapd # LabelValueScaled
import visualization as viz
# can potentially improve the speed of numpy value remapping using:
# https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
# k = np.array(list(mapping.keys()))
# v = np.array(list(mapping.values()))

# mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) #k,v from approach #1
# mapping_ar[k] = v
# out = mapping_ar[input_array]


# from importlib import reload
# import numpy as np
# import torch
# import wandb
# import pytorch_lightning as pl
# import monai
# from monai.data import decollate_batch

#%% need to setup: os.environ["DATA_ROOT"]
# e.g.: os.environ["DATA_ROOT"] = f"/project/rrg-mfbeg-ad/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/UBCMIXDEM_WMHT1T2relationships"

#%% 
def get_image_label_paths(cfg):
  '''prepare data module
  # Test suit:
  # prepare_data_module(cfg)
  '''
  ## prepare all data types
  # root_dir = Path(cfg["root_dir"])
  root_dir = os.getenv("DATA_ROOT")

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

  ## Only deal with single-input-type only
  #  determine what "image_modality" to use
  if cfg["image_type"] == "T1":
    image_paths = image_paths_t1
  elif cfg["image_type"] == "T2":
    image_paths = image_paths_t2
  #  determine what "segmentation_types" to use
  if cfg["seg_type"] == "WM":
    label_paths = label_paths_wm
  elif cfg["seg_type"] == "T1":
    label_paths = label_paths_t1
  elif cfg["seg_type"] == "T2":
    label_paths = label_paths_t2
  ## To deal with multi-input-types later

  assert len(image_paths) > 0 # ensure there are files
  assert len(image_paths) == len(label_paths)

  return image_paths, label_paths

# %% prepare data module
def create_data_module(cfg, initialize=True, verbose=True):
  ''' Create DataModule for segmentation tasks 
  - file location stored in: data.data_dicts'''
  if verbose > 0: print('  >>> creating DataModule...')
  
  # get image label paths
  image_paths, label_paths = get_image_label_paths(cfg)
    
  old_labels = cfg["raw_labels"]
  batch_size_train = cfg["batch_size_train"]
  batch_size_val   = cfg["batch_size_val"]
  patch_size = tuple(cfg['patch_size'])
  new_labels = list(range(1,len(old_labels)+1))
  label_transformations = [LabelValueRemapd(keys=['label'], old_labels=old_labels, new_labels=new_labels, zero_other_labels=True)]
  
  data = datasets.DataModuleWMH(image_paths, label_paths, patch_size=patch_size, label_transformations=label_transformations, batch_size_train=batch_size_train, 
  batch_size_val=batch_size_val, 
  dataset_type="CacheDataset", num_workers=0)
  
  if initialize:
    data.setup()

  return data

# %% prepare model module
def create_model(cfg, model_type="unet_3d", verbose=True):
  '''create model module
  - To check learning rate: print(unet_wms.lr)
  '''
  if verbose > 0: print('  >>> creating unet_3d model...')

  lr          = float(cfg['lr']) if 'lr' in cfg.keys() else 1e-3
  in_channels = cfg['in_channels'] if 'in_channels' in cfg.keys() else 1
  label_num   = cfg['label_num']   # 10  # = len(lbl_b.unique())
  device      = cfg['device'] if 'device' in cfg.keys() else 'cuda'
  model_type  = cfg['model_type'] if 'model_type' in cfg else 'unet_3d'
  
  if model_type == "unet_3d":
    # number of residual units needed for the unet
    num_res_units = cfg['num_res_units'] if 'num_res_units' in cfg.keys() else 2
    net = models.unet_3d(in_channels=in_channels,
                        out_channels=label_num,
                        num_res_units=num_res_units)
  unet_wms = models.UNet3DModule(net=net, lr=lr, device=device)
  
  return unet_wms

#%%  define trainer
def create_trainer(cfg, verbose=True):
  '''create trainer module'''
  if verbose > 0: print('  >>> creating trainer ...')

  root_dir   = Path(cfg["root_dir"])
  max_epochs = cfg["max_epochs"] if 'max_epochs' in cfg.keys() else 1000
  # for `ModelCheckpoint` callback to save the best performances model
  monitor    = cfg["monitor"] if 'monitor' in cfg.keys() else "val_dice"
  monitor_mode = cfg["monitor_mode"] if "monitor_mode" in cfg.keys() else "max" # save the max monitored "val_dice"
  save_top_k = cfg["save_top_k"] if "save_top_k" in cfg.keys() else 1

  now = datetime.now()
  current_time = f"{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}:{now.second}"
  exp_name   = f"{cfg['exp_name']}_{current_time}" if "exp_name" in cfg.keys() else current_time
  
  processed_dir = root_dir/"PROCESSED_DATA"
  processed_dir.mkdir(exist_ok=True, parents=True)
  # checkpoint_dir = processed_dir/"checkpoints"
  
  trainer = train.lightning_trainer(default_root_dir=processed_dir, 
                                    log_dir=f"{processed_dir}/logs/{exp_name}",
                                    max_epochs=max_epochs, 
                                    exp_name=exp_name, 
                                    monitor=monitor,
                                    mode=monitor_mode,
                                    save_top_k=save_top_k)
  return trainer

#%% core training function
def train_model(yaml_file, data=None, model=None, trainer=None):
  ''' run model training according to yaml_file configuration
  yaml_file = "_wm_segs/_wm_segs_local_linux.yaml"
  yaml_file = "_wm_segs/_wm_segs_local_windows.yaml"
  
  - May provide the data module to speed up initialization'''
  # %%
  # initialization (also add the library directory to the python path)
  cfg = utils.read_yaml(yaml_file)

  # create model
  if model is None:
    model = create_model(cfg)

  # create trainer
  if trainer is None:
    trainer = create_trainer(cfg)

  # prepare data module
  if data is None:
    data = create_data_module(cfg)

  # %%
  # Start training with learning rate = {lr}
  print(f'  >>> Start training with learning rate = {model.lr}...')
  trainer.fit(model, data)

  # %%
  # save the model
  exp_name = cfg["exp_name"]
  print('  >>> saving the best model...')
  checkpoint_save_path = f"{trainer.default_root_dir}/saved_models/{exp_name}_best_model.ckpt" # try to also add dice score to the model name
  trainer.save_checkpoint(checkpoint_save_path)
  
  return {"data": data, "model": model, "trainer": trainer, "checkpoint_save_path": checkpoint_save_path}

#%% for developing/debugging purpose
def test_suit():
  # %%
  yaml_file = "_wm_segs/_wm_segs_remote.yaml"
  # ckpt_path = "/home/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/saved_models/UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_1e-3_best_model.ckpt"
  ckpt_path = glob("/home/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/UBCMIXDEM_WMHT1T2relationships/PROCESSED_DATA/checkpoints/*.ckpt")[0]
  data=None
  model=None
  trainer=None
  eval_mode="whole"
  val_id=0
  device='cuda'
  verbose=True
  quickcheck=True

# %% model inference function (using sliding window inference)
def model_inference(yaml_file, ckpt_path, input_paths, output_dir, pixdim=(1.0, 1.0, 1.2), axcodes="RAS", device='cuda', batch_size=1, verbose=True, sw_batch_size=1, overlap=0.25, quickcheck=False, output_postfix='', output_ext='.nii.gz', data_root_dir=''):
  '''segmentation model inference
  - yaml_file: yaml file used to create the model and transformations.
  - ckpt_path: model checkpoint path
  - input_path: path of the input (nifti) file
  - data_root_dir: input paths data root 
  # Ref: https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference
  '''

  # %%
  if not os.path.isfile(ckpt_path): 
      raise FileNotFoundError(f"{ckpt_path} does not exist")
  
  #%% read yaml_file
  cfg = utils.read_yaml(yaml_file)
  
  # %% 
  # create test/inference transformations
  test_transforms = t.Compose([
    # deterministic
    t.LoadImaged(keys=["image"]),
    t.AddChanneld(keys=['image']),
    t.Spacingd(keys=['image'],pixdim=pixdim, mode=("bilinear")),
    t.Orientationd(keys=["image"], axcodes=axcodes),
    t.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    t.ToTensord(keys=["image"]),
    t.EnsureTyped(keys=["image"], data_type='tensor'),
  ])

  # %%
  if not isinstance(input_paths, list):
    input_paths = [input_paths]
  # if not isinstance(output_paths, list):
  #   output_paths = [output_paths]
  #  create test data dict
  test_data_dicts = [{"image": str(input_path)} for input_path in input_paths]
  test_dataset = Dataset(test_data_dicts,transform=test_transforms)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

  # %%
  # check shape
  # Alternative straightforward way to create single test data
  # test_data = test_transforms(test_data_dicts)
  # test_data[0]['image'].unsqueeze(0).to(device).shape

  # %% 
  # create model
  model = create_model(cfg)

  # %% 
  # input model weight from ckpt_path
  if verbose > 0: 
    print(f"  >>> loading model from {ckpt_path}")
  model.load_from_checkpoint(checkpoint_path=ckpt_path, net=model._model, map_location=device, strict=False, verbose=verbose);
  # same as: model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'], strict=False, verbose=verbose)
  model.eval()
  model.to(device);
  # %%
  # inference
  for test_batch in test_dataloader:
    img = test_batch['image'].to(device)
    print(img.shape)
    test_outputs = sliding_window_inference(inputs=img, roi_size=cfg['patch_size'], sw_batch_size=sw_batch_size, predictor=model, overlap=overlap)
    # %%
    # display segmentation results for the first on ein the batch
    batch_id = 0
    if quickcheck == True:
      vol = img[batch_id].detach().cpu().squeeze()
      seg = test_outputs.argmax(dim=1)[batch_id].detach().cpu().squeeze()
      viz.vol_peek(vol=vol, overlay_vol=seg)
    
    # %% 
    
    # save the output segmentation 
    batch_vol = test_outputs.argmax(dim=1, keepdims=True)
    utils.save_batch_to_nifti(batch_vol, meta_data=None, output_dir=output_dir, output_postfix=output_postfix, output_ext=output_ext, resample=False, squeeze_end_dims=True, data_root_dir=data_root_dir, separate_folder=False, print_log=True)
    # break


# %% core model validation function
def evaluate_model(yaml_file, ckpt_path=None, data=None, model=None, eval_mode="whole", val_id=0, device='cuda', quickcheck=True, verbose=True):
  '''model evaluation
  - visualize sample segmentation results from a batch
  - write segmentation dice score (Either on title or separately) 

    yaml_file = "_wm_segs/_wm_segs_local_linux.yaml"
    yaml_file = "_wm_segs/_wm_segs_local_windows.yaml"
    yaml_file = "_wm_segs/_wm_segs_remote.yaml"
    
  - ckpt_path = os.path.join(trainer.default_root_dir,"saved_models/UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_1e-3_best_model.ckpt")
    # ckpt_path = 'g:\\My Drive\\Data\\Brain_MRI\\T1_Hypo_T2_Hyper\\UBCMIXDEM_WMHT1T2relationships\\PROCESSED_DATA\saved_models\\UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_1e-3_best_model.ckpt'
  - eval_mode:
    - "crop": only plot regions acound the bbox of segmented regions (better for visualing smaller leision segmentation)
    - "whole": through the sliding window approach (better for visualing whole brain/WM segmentations)
  
  # Ref: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
  # Ref: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb
  https://docs.monai.io/en/latest/inferers.html?highlight=sliding_window_inference#sliding-window-inference
  #   '''
  # %%
  from monai.inferers import sliding_window_inference
  import visualization as viz

  cfg = utils.read_yaml(yaml_file)

  device = cfg['device'] if 'device' in cfg.keys() else 'cuda'

  # %%
  #  prepare data module
  if data is None:
    data = create_data_module(cfg)

  # %%
  if model is None:
    #  create model (lightning module)
    model = create_model(cfg)

    # %%
    # load pretrained model weights
    # Ref: https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html#weights-loading
    if not os.path.isfile(ckpt_path): 
      raise FileNotFoundError(f"{ckpt_path} does not exist")
    else:
      if verbose > 0: print(f"  >>> loading model from {ckpt_path}")
      model.load_from_checkpoint(checkpoint_path=ckpt_path, net=model._model, map_location=device, strict=False);
      # model.load_state_dict(torch.load(ckpt_path))
      model.eval()
      model.to(device);

    # %%
    for val_id in range(len(data.valid_data)):
      # get one bath of the validation data (in total only two)
      val_batch = data.valid_data[val_id]
      # add the batch channel
      img = val_batch['image'].unsqueeze(0).to(device)
      lbl = val_batch['label'].unsqueeze(0).to(device)
      print(img.shape, lbl.shape)

      # %%
      # sliding window inference
      sw_batch_size = cfg['sw_batch_size'] if 'sw_batch_size' in cfg.keys() else 1 # default inference 1 batch of patch at a time
      overlab = cfg['sw_overlap_ratio'] if 'sw_overlap_ratio' in cfg.keys() else 0.25 # default have 0.25 overlap between consequtive adjacent patches
      val_outputs = sliding_window_inference(inputs=img, roi_size=cfg['patch_size'], sw_batch_size=6, predictor=model, overlap=0.25)

      # %%
      # display segmentation results
      if quickcheck == True: 
        vol = img.detach().cpu().squeeze()
        seg = val_outputs.argmax(dim=1).detach().cpu().squeeze()
        viz.vol_peek(vol=vol, overlay_vol=seg)
      # %%





      # %%
      break
  
  # %%
  return

#%%
def arg_parse():
  '''
  Parse input arguments
  `nargs='?'`: means 0-or-1 arguments
  # Ref: https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
  '''
  parser = argparse.ArgumentParser(description='Train a UNet model on WMH data')
  parser.add_argument('-y', '--yaml_file', dest='yaml_file',
                      help='yaml file',
                      default=None, type=str)
  parser.add_argument('-p','--ckpt_path', dest='ckpt_path',
                      help='model path (for evaluation purpose)',
                      default=None, type=str)
  parser.add_argument('-m', '--mode', dest='mode', nargs='?', 
                      help='running mode (training/evaluation)',
                      default='evalu', type=str)
  parser.add_argument('-e', '--eval_mode', dest='eval_mode', nargs='?', 
                      help='evaluation mode (training/evaluation)',
                      default='whole', type=str)
  args = parser.parse_args()
  return args

#%%
def main():
  # %% 
  # read input arguments
  # debugging line overriding sys.argv
  # Ref: https://stackoverflow.com/questions/50886471/simulating-argparse-command-line-arguments-input-while-debugging
  # sys.argv = ['', '-y', '_wm_segs/_wm_segs_remote.yaml','-p', 'g:\\My Drive\\Data\\Brain_MRI\\T1_Hypo_T2_Hyper\\UBCMIXDEM_WMHT1T2relationships\\PROCESSED_DATA\saved_models\\UBC_UNet_WhiteMatterSegmentation_DiceCE_lr_1e-3_best_model.ckpt']
  # print(sys.argv)
  args = arg_parse()
  yaml_file = args.yaml_file
  ckpt_path = args.ckpt_path
  mode = args.mode
  
  for key, value in args._get_kwargs():
    print(f"{key}: {value}")
  
  # %%
  if mode == "train":
    print("running model training")
    train_model(yaml_file)
  elif mode == "eval":
    print("running model evaluation")
    evaluate_model(yaml_file, ckpt_path, mode)

#%%
if __name__ == '__main__':
  # %%
  main()  
  
  # %%
  # yaml_file = "_wm_segs/_wm_segs_local_linux.yaml"
  
  # %%
  # define yaml file
  # yaml_file  = sys.argv[-1]

  # %%
  # run training
  # train_model(yaml_file)





# %%
