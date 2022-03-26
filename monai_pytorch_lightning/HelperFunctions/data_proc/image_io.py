

#%%
def resize(vol, size, channel=None, interpolation=3):
  '''2D resizing numpy array / torch tensor / PIL image
  - Input
    - vol: numpy array or torch tensor
    - size: output size
    - channel: 'fist'/'last'
      - numpy array: default last
      - PIL image: default last
      - torch tensor: default first
  Ref:
  - resize numpy of arbitrary dimension:
  https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image/13251340
  > Bilinear interpolation would be order=1, nearest is order=0, and cubic is the default (order=3).
  - resize torch.Tensor (F.interpolate)
  '''
  import numpy as np, scipy, torch, PIL
  # convert single size to height/width
  if type(size) is int:
    size = (size, size)
  # type-specific resizing
  if type(vol) is np.ndarray:
    # order: 0 nearest; 3: bilinear?
    return scipy.ndimage.zoom(vol,(size[-2]/vol.shape[-2],size[-1]/vol.shape[-1]), order=interpolation)
    ##  or use: (Ref: openMultiChannelImage)
    # from skimage.transform import resize
    # vol = resize(vol, (size,size))
  elif type(vol) is PIL.Image.Image:
    return vol.resize((size[0], size[1]))
  elif type(vol) is torch.Tensor:
    # mode (str): algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` | ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    import torch.nn.functional as F
    vol = F.interpolate(vol.permute(0,2,1), size[0], mode=interpolation)
    vol = F.interpolate(vol.permute(0,2,1), size[1], mode=interpolation)
    return vol