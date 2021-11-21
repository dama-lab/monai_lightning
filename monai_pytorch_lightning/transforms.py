#%% import libraries
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection, NdarrayTensor
import monai.transforms as t
from monai.transforms import Transform, MapTransform
from monai.transforms.spatial.dictionary import * # A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations defined in :py:class:`monai.transforms.spatial.array`.
from pathlib import Path
import numpy as np

class LabelValueScaled(MapTransform):
    """
    LabelValueScaled class (ref: AddChanneld from MONAI)
    > To rescale the value of the label (e.g. scale=1/1000 will scale lable from 1000 to 1)
    
    # Test Suite
    # T = LabelValueScaled(keys=['label'],scale=1) # 1/1000 for T2 Hyper-Intensity
    """

    def __init__(self, keys: KeysCollection, 
                scale: float=1,
                allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.scale = scale

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key])*self.scale
        return d

def label_value_remap(img: np.ndarray, 
                    old_labels: Sequence[int] = [], 
                    new_labels: Sequence[int] = [], 
                    zero_other_labels: bool = True) -> np.ndarray:
    """ Remap label values (independent function version)
    - zero_other_labels:
        - if True: non-remapped voxels will be assigned zero labels
        - if False: non-remapped voxels will retain original label values 
    # test_suilt: 
    """
    if zero_other_labels == True:
        img_new = np.zeros(img.shape)
    else:
        img_new = np.copy(img)

    for i, lbl in enumerate(old_labels):
        img_temp = np.zeros(img.shape)
        img_new[img==lbl] = 0 # zero the voxels-to-be-remapped first
        img_temp[img==lbl] = new_labels[i]
        img_new = img_new + img_temp
    return img_new

class LabelValueRemap(Transform):
    def __init__(self, 
                old_labels: Sequence[int] = [], 
                new_labels: Sequence[int] = [], 
                zero_other_labels: bool = True) -> None:
        """ Remap label values (array version, compatible with MONAI)
        """
        # if only one label is given, convert the label to list
        if type(old_labels) is int: old_labels = [old_labels]
        if type(new_labels) is int: new_labels = [new_labels]
        self.old_labels, self.new_labels = old_labels, new_labels
        self.zero_other_labels = zero_other_labels

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> List[Union[np.ndarray]]: # potentially extend it to work with tensor in the future
        return label_value_remap(img, self.old_labels, self.new_labels, self.zero_other_labels)

class LabelValueRemapd(MapTransform):
    """ Remap label values (dictionary version, compatible with MONAI)
    """

    def __init__(self, keys: KeysCollection,
                old_labels: Sequence[int] = [], 
                new_labels: Sequence[int] = [], 
                zero_other_labels: bool = True,
                allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.remapper = LabelValueRemap(old_labels=old_labels, new_labels=new_labels, zero_other_labels=zero_other_labels)

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.remapper(d[key])
        return d

# %% for retina
def select_retinal_region(x, bg_bthre=0, bg_uthre=8):
  '''based on label only, can create one based on image later'''
  return ((x > bg_bthre) & (x < bg_uthre))

class MoveToDevice(Transform):
    def __init__(self, device = 'cuda') -> None:
        """ Remap label values (array version, compatible with MONAI)
        """
        # if only one label is given, convert the label to list
        self.device = device

    def __call__(self, data: NdarrayTensor) -> NdarrayTensor:
        # def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> List[Union[np.ndarray]]: # potentially extend it to work with tensor in the future
        return data.to(self.device)

class MoveToDeviced(MapTransform):
    """ Remap label values (dictionary version, compatible with MONAI)
    """

    def __init__(self, keys: KeysCollection, device = 'cuda', allow_missing_keys: bool = True) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mover = MoveToDevice(device=device)

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mover(d[key])
        return d

# %% setup transforms for training and validation
def transforms_3d(pixdim=(1.0, 1.0, 1.2), patch_size=(96, 96, 96), axcodes="RAS", label_transformations=[]):
  '''
  - Sample `label_transformations` function:
    - T1 leision label: `LabelValueScaled(keys=['label'],scale=1)` # doing nothing
    - T2 leision label: `LabelValueScaled(keys=['label'],scale=1/1000)`
    - T1 parcellation label: need to remap, not yet implemented 
    - get nifty pixel dimension using: nib.load(*).header.get_zooms()
  '''
  train_transforms = t.Compose([
    # deterministic
    t.LoadImaged(keys=["image","label"]),
    t.AddChanneld(keys=["image"]),
    t.Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear", "nearest"),),
    t.Orientationd(keys=["image", "label"], axcodes=axcodes),
    # scale label value (maxmum value of the current dataset is: 2769630.8 (float32))
    # t.ScaleIntensityRanged(keys=["image","label"], a_min=0, a_max=2800000, b_min=0.0, b_max=1.0, clip=True),
    # CropForeground by default create bbox that exclude surrounding zero pixels in the "image"
    t.CropForegroundd(keys=["image", "label"], source_key="image"),
    t.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    *label_transformations,
    # LabelValueScaled(keys=['label'],scale=1),
    
    # random transformation
    # randomly crop out patch samples from big image based on pos / neg ratio
    # the image centers of negative samples must be in valid image area
    t.RandCropByPosNegLabeld(
      keys=["image", "label"],
      label_key="label",
      spatial_size=patch_size,
      pos=1,
      neg=1,
      num_samples=4,
      image_key="image",
      image_threshold=0,),
    t.RandScaleIntensityd(keys='image', factors=0.1, prob=0.5),
    t.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.5),
    t.RandAffined(
      keys=["image","label"],
      mode=("bilinear","nearest"),
      prob=0.5,
      spatial_size=patch_size,
      rotate_range=(0, 0, ),
      scale_range=(0.1,0.1,0.1),
    ),
    t.ToTensord(keys=["image","label"]),
    t.EnsureTyped(keys=["image", "label"], data_type='tensor'),
  ])
  val_transforms = t.Compose([
    # deterministic
    t.LoadImaged(keys=["image","label"]),
    t.AddChanneld(keys=['image','label']),
    t.Spacingd(keys=['image','label'],pixdim=pixdim, mode=("bilinear", "nearest"),),
    t.Orientationd(keys=["image", "label"], axcodes=axcodes),
    # t.Resized(keys=["image","label"], spatial_size=val_resize)
    t.CropForegroundd(keys=["image", "label"], source_key="image"), # might cause validation dataset to be different
    # crop the center region
    # t.CenterSpatialCropd(keys=["image","label"], roi_size=patch_size),
    # scale label value
    t.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    *label_transformations,
    t.ToTensord(keys=["image","label"]),
    t.EnsureTyped(keys=["image", "label"], data_type='tensor'),
  ])
  test_transforms = t.Compose([
    # deterministic
    t.LoadImaged(keys=["image"]),
    t.AddChanneld(keys=["image"]),
    t.Spacingd(keys=["image"],pixdim=pixdim, mode=("bilinear", "nearest"),),
    t.Orientationd(keys=["image"], axcodes=axcodes),
    # t.Resized(keys=["image"], spatial_size=val_resize)
    t.CropForegroundd(keys=["image"], source_key="image"), # might cause validation dataset to be different
    # crop the center region
    # t.CenterSpatialCropd(keys=["image"], roi_size=patch_size),
    # scale label value
    t.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    *label_transformations,
    t.ToTensord(keys=["image"]),
    t.EnsureTyped(keys=["image"], data_type='tensor'),
  ])
  return train_transforms, val_transforms, test_transforms

# %% for 2D convolutional neural network
def transforms_2d(patch_size=(512,512,1), rotate90_spatial_axes=(0,1), rotate90_k=3, crop_forground_source_key="label", crop_forground_select_fn=None, label_transformations=[]):
  '''
  setup transforms
  '''
# %% === deterministic ===
  def set_deterministic_transforms(keys=["image","label"], rotate90_spatial_axes=rotate90_spatial_axes, rotate90_k=rotate90_k, crop_forground_source_key=crop_forground_source_key, crop_forground_select_fn=crop_forground_select_fn):
    '''setup deterministic transforms
    rotate90_k: time to perform rotate90
    '''
    deterministic_transforms = [
      t.LoadImaged(keys=keys),
      t.AddChanneld(keys=keys),
      *label_transformations,
    ]
    # add rotate if necessary
    if rotate90_spatial_axes is not None:
      deterministic_transforms.append(
        t.Rotate90d(keys=keys, k=rotate90_k, spatial_axes=rotate90_spatial_axes,)
        )
    if crop_forground_select_fn is not None:
      deterministic_transforms.append(
        t.CropForegroundd(keys=keys, source_key=crop_forground_source_key, select_fn=crop_forground_select_fn)
      ) # select_retinal_region
    deterministic_transforms.append(t.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),)
    return deterministic_transforms
    
  # === random transform ===
  random_transform = [
    t.RandCropByPosNegLabeld(
      keys=["image","label"],
      label_key="label",
      spatial_size=patch_size,
      pos=1,
      neg=1,
      num_samples=4,),
    t.SqueezeDimd(keys=["image","label"],dim=-1),
    t.RandScaleIntensityd(keys='image', factors=0.1, prob=0.5),
    t.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.5),
    t.RandAffined(
      keys=["image","label"],
      mode=("bilinear","nearest"),
      prob=0.5,
      spatial_size=patch_size[:2],
      rotate_range=(0,),
      scale_range=(0.1,0.1),
      ),
  ]

  # ===== train transform =====
  train_transforms = t.Compose([
    *set_deterministic_transforms(keys=["image","label"]),
    *random_transform,
    t.ToTensord(keys=["image","label"]),
    t.EnsureTyped(keys=["image", "label"], data_type='tensor'),
  ])

  val_transforms = t.Compose([
    *set_deterministic_transforms(keys=["image","label"]),
    t.ToTensord(keys=["image","label"]),
    t.EnsureTyped(keys=["image", "label"], data_type='tensor'),
    MoveToDeviced(keys=["image", "label"], device='cpu')
  ])

  test_transforms = t.Compose([
    *set_deterministic_transforms(keys=["image"]),
    t.ToTensord(keys=["image"]),
    t.EnsureTyped(keys=["image"], data_type='tensor'),
    MoveToDeviced(keys=["image"], device='cpu'),
  ])

  return train_transforms, val_transforms, test_transforms
