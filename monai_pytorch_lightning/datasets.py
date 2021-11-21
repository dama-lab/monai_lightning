# %% load libraries
import os, sys
from monai.data.utils import pad_list_data_collate # , list_data_collate
import pytorch_lightning as pl

from monai.data import DataLoader
from pathlib import Path

from monai.data import CacheDataset, PersistentDataset, Dataset
from monai.utils.misc import set_determinism
import monai.transforms as t

# import torch
# import monai
# from transforms import LabelValueScaled


#%% DataModule class template
class DataModule_template(pl.LightningDataModule):
  def __init__(self):
    # define required parameters here
    return
  def prepare_data(self):
    # define steps done only on GPU, e.g getting data
    return
  def setup(self, stage=None):
    # define steps done on every GPU, e.g. split data, apply transform
    return
  def train_dataloader(self):
    return
  def val_dataloader(self):
    return
  def test_dataloader(self):
    return
  def teardown(self):
    # clean up after fit or test, called on every process in DDP
    return
        
#%% DataModule class

class DataModule(pl.LightningDataModule):
  def __init__(self, image_paths, label_paths, train_transforms, val_transforms, test_transforms=None, batch_size_train=32, batch_size_val=1, batch_size_test=1, dataset_type='CacheDataset', num_workers=None, num_workers_cache=None, cache_rate=1.0, tmp_dir='/tmp/dma73/', collate_fn=pad_list_data_collate):
    '''
    - label_paths can be None to facilitate test dataloader
    - train_files, val_files [data_dict]: {"image": str(image_name), "label": str(label_name)}
    - dataset_type = CacheDataset (better for debugging); PersistentDataset (better for deloy, cannot reflect changes during development after been initialized and cached)
    ------
    - Example of setting up data path:
      ```python
      root_dir = Path("/project/6003102/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/")
      processed_dir = root_dir/"PROCESSED_DIR"
      processed_dir.mkdir(exist_ok=True, parents=True)
      images_dir = Path(f"{root_dir}/RAW_DATA/")
      # T1
      image_paths = sorted(images_dir.glob("*_T1W.nii.gz"))
      # T2
      image_paths_T2 = sorted(images_dir.glob("*_T2WFLAIRinT1W.nii.gz"))
      # T1 label
      label_paths_t1 = sorted(images_dir.glob("*_T1HYPOWMSAinT1W.nii.gz"))
      # T2 label
      label_paths = sorted(images_dir.glob("*_T2HYPERWMSAinT1W.nii.gz"))
      assert len(image_paths) == len(label_paths)
      ```

    - Sample `label_transformations` function:
      - T1 leision label: `LabelValueScaled(keys=['label'],scale=1)` # doing nothing
      - T2 leision label: `LabelValueScaled(keys=['label'],scale=1/1000)` # divide by 1000
      - T1 parcellation label: need to remap, not yet implemented 
    '''
    # define required parameters here
    super().__init__()
    self.download_dir = tmp_dir
    self.batch_size_train = batch_size_train
    self.batch_size_val   = batch_size_val
    self.batch_size_test  = batch_size_test
    
    self.train_transforms = train_transforms
    self.val_transforms   = val_transforms
    self.test_transforms   = test_transforms

    if num_workers_cache is None: num_workers_cache = os.cpu_count()
    if num_workers is None: num_workers =  os.cpu_count()
    self.num_workers_cache = num_workers_cache
    self.num_workers = num_workers
    self.collate_fn  = collate_fn
    self.dataset_type = dataset_type
    self.cache_rate  = cache_rate
    # create persistant_cache_dir with cache_root = tmp_dir
    self.persistent_cache_dir = Path(tmp_dir,'persistent_cache')
    self.persistent_cache_dir.mkdir(parents=True, exist_ok=True)
    self.image_paths, self.label_paths = image_paths, label_paths

  def prepare_data(self):
    # define steps done only on GPU, e.g getting/downloading data
    return

  def setup(self, val_idx=-2, stage=None):
    # define steps done on every GPU, e.g. split data, apply transform on dataset
    # Question: whether the part below shall be in prepare_data or setup?
    set_determinism(seed=0)

    # setup data path
    self.data_dicts = [{"image": str(image_name), "label": str(label_name)} for image_name, label_name in zip(self.image_paths, self.label_paths)]

    self.train_files, self.val_files = self.data_dicts[:val_idx], self.data_dicts[val_idx:]

    if self.dataset_type == "CacheDataset":
      # training data
      self.train_data = CacheDataset(data=self.train_files, transform=self.train_transforms, num_workers=self.num_workers_cache, cache_rate=self.cache_rate)
      # validation data
      self.valid_data = CacheDataset(data=self.val_files, transform=self.val_transforms, num_workers=self.num_workers_cache, cache_rate=self.cache_rate)
    elif self.dataset_type == "PersistentDataset":
      # (for final production)
      # training data
      self.train_data = PersistentDataset(data=self.train_files, transform=self.train_transforms, cache_dir=self.persistent_cache_dir)
      # validation data
      self.valid_data = PersistentDataset(data=self.val_files, transform=self.val_transforms, cache_dir=self.persistent_cache_dir)

    # self.test_data not yet defined

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=self.num_workers) # collate_fn=self.collate_fn, 
    
  def val_dataloader(self):
    return DataLoader(self.valid_data, batch_size=self.batch_size_val, shuffle=False, num_workers=self.num_workers, pin_memory=True) # collate_fn=self.collate_fn, 
    
  def test_dataloader(self):
    test_data_dicts = [{"image": str(input_path)} for input_path in self.image_paths]
    # use base Dataset class for testing
    self.test_dataset = Dataset(test_data_dicts,transform=self.test_transforms)
    return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=0)





        
# %%
