#%% import libraries
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection, NdarrayTensor
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

        
