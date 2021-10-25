# -*- coding: utf-8 -*-

# Create customized dataset

## > Ref: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# Pytorch Official:
  # > https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py
  # > https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
  # > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

#%% Custom Dataset Fundamentals
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformation
def albumentations_transform(size=(512,512),internal_size=(224,224)):
  # torchvision version:
  # transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
  # albumentation version:
  transformation = A.compose([
    A.Resize(*size),
    A.RandomCrop(*internal_size),
    A.HorizontalFlip(),
    A.Normalize(
      mean= ([0.485,0.456,0.406]),
      std =([0.229,0.224,0.225]),
      ),
    ToTensorV2()
    ])
  return transformation

class MyCustomDataset(Dataset):
  
  def __init__(self, transforms=None):
    '''
    __init__() function is where the initial logic happens like reading a csv, assigning transforms, filtering data, etc.   
        
    Parameters
    ----------
    transforms :
    
    '''
    super().__init__()
    
    # transformation
    self.transforms = albumentations_transform
    # stuff
    
  def __getitem__(self, index):
    ''' __getitem__() function returns the data and labels. 

    Parameters
    ----------
    index :

    Returns
    -------
    batch must contain tensors, numbers, dicts or lists
    '''
    # stuff
    data
    
    data = self.transformations(data)
    
    return (img, label)
  
  def __len__(self):
    return count # number of how many examples (images) you hav
  
  
# %% Data Loader (from MONAI)
# > ref: https://colab.research.google.com/drive/1e8hoI12FXEq9I9OPzLyLiKaiBmcTXWZz#scrollTo=j2M_vInWe6A_
def MyDataLoader(DataSet, batch_size=32, shuffle=True):
  my_data_loader = DataLoader(dataset=DataSetm
                              batch_size=batch_size,
                              shuffle=shuffle)
  