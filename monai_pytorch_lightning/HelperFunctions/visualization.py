# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_HelperFunctions.visualization.ipynb (unless otherwise specified).

__all__ = ['show_label', 'fig2img', 'removeBackground', 'rescaleIntensity', 'vol_peek', 'quickcheck_generator',
          'untar_quickcheck_generator']

# Cell
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch, gc
from fastai.vision import *

from ..data_io import image_io
from .helper_functions import *
# Cell

# %%%%%%%%%%%%%%%%%%% QuickCheck Tensor Image %%%%%%%%%%%%%%%%%%%%%
def show_label(img:torch.Tensor, lbl:torch.Tensor):
  ''' overlay a tensor lable to tensor img
  - Input:
    - img:$FAISAL_MANLBLQCTOOLS_DIR/data/tissue-segmentation/TargetLists/vertebral_segmentation/TargetList-round_3_for_training
  '''

  ### remove label background
  # convert integer to float
  lbl = lbl.float()
  # convert 0 (backgroun) to NaN (Ref: https://github.com/pytorch/pytorch/issues/4767)
  lbl[lbl==0] = torch.tensor(float('nan'))

  img = Image(img)
  lbl = Image(lbl)
  img.show(y=lbl)

# Cell

def fig2img(fig, mode='RGB', format='png', dpi=300, bbox='tight'):
  ''' convert matplotlib.plot into PIL.image
  -Input:
    - mode = ['RGB','RGBA']
  # Ref: https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
  # Ref: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
  # Ref: https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
  '''

  import io
  from PIL import Image

  buf = io.BytesIO()
  fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
  buf.seek(0)
  img = Image.open(buf)

  if mode is not None:
    img = img.convert(mode)

  # close buffer
  buf.close() # or: buf.truncate(0); # = buf.seek(0); buf.trancate()
  # Alternative: # figImg = Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
  return img

# Cell
# Helper function
def removeBackground(labelVol, bgLabel=0, **kwargs):
    '''remove background voxels in the label (change to nan)
    - input:
        - labelVol: numpy array
        - bgLabel:  background label number, (default=0)
    - return:
        - labelVol: numpy array
    '''
    # convert label into float type (so that can assign NaN to it, as NaN is a float)
    labelVol = labelVol.astype(float)
    # convert background labels to Nan
    labelVol[labelVol==bgLabel] = np.nan

    return labelVol

# Cell
def rescaleIntensity(vol, threRange=None, rescaleTo=None):
  ''' rescale/threshold volume intensity according to intensity range
  - Input
    - vol:
    - threRange: thresholds range e.g.  [-190 150] for CT
    - rescaleTo: (CT=[0,255])
  # Ref:  # function [img] = abacsfem_map_CTimage_to_256(img,img_msk,range)
  '''
  if rescaleTo == None:
    rescaleTo = [vol.min(), vol.max()]
  if threRange == None:
    ratio = 0.1
    eliminate = (rescaleTo[1] - rescaleTo[0])*ratio
    threRange = [rescaleTo[0]+eliminate, rescaleTo[1]-eliminate]

  volScaled = vol.copy()
  tempVol = vol.copy()

  volScaled[ tempVol<threRange[0] ] = rescaleTo[0]
  volScaled[ tempVol>threRange[1] ] = rescaleTo[1]

  temp_idx = ( (tempVol >= threRange[0]) + (tempVol <= threRange[1]) )
  LOW = min(tempVol[temp_idx])
  HIGH = max(tempVol[temp_idx])
  volScaled[temp_idx] = np.ceil(255*(volScaled[temp_idx]-LOW)/(HIGH-LOW))
  img_msk = (volScaled > 0)
  volScaled[img_msk != 1] = 0

  return volScaled

################
# %% 2D %%%%%%%%%%%%%%%%%%%%%

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def slice_viewer(X):
  ''' Interactive slice-through 3D volume viewer
  > Ref https://matplotlib.org/gallery/animation/image_slices_viewer.html


  Parameters
  ----------
  vol : TYPE
    DESCRIPTION.

  Returns
  -------
  None.

  '''

  fig, ax = plt.subplots(1, 1)
  
  tracker = IndexTracker(ax, X)
  
  
  fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
  plt.show()
  
  return fig



# Cell
# %% 3D %%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%% QuickCheck %%%%%%%%%%%%%%%%%%%%%
def vol_peek(vol:tuple([np.array,torch.Tensor,Path,str]),
            rows=5, cols=5,
            filepath=None, fig_title=None, permute_axes=None, transpose=False, flipud=False,
            rescale=False, threRange=None, rescaleTo=None, overlay_vol:tuple([np.array, torch.Tensor,Path,str])=None, overlay_surf=None, alpha=0.7,
            volThre=None, volCmap='gray', labelCmap='jet', labelNorm=None,
            volClim=None, labelClim=None,
            sliceNo=True, figsize=16, dpi=300, aspect='auto', close_figure=False, show=True, returnType='figure', **kwargs):
    ''' To generate a panaroma/montage peak image of a numpy volume
    - Input:
        vol (np): volumetric numpy array
        rows (int): number of rows in the plot
        cols (int): number of columes in the plot
        filepath (str/Path): location to save montage
        fig_title:
        permute_axes: (higher priority than transpose)
        transpose: (default=True) swap 0th and 1st dimension for OCTvisualization (only when permute_axes undefined)
        rescale: contrast enhance
        threRange: (CT = [-190, 150])
        rescaleTo: (ct = [0,255])
        flipud
        overlay_vol:
        overlay_surf: [width,depth,layer]
        alpha: transparency of the overlay image/contour
        volThre: volume threshold (e.g. [-190 150] for ct)
        volCmap:
        labelCmap:
        volClim:
        labelClim:
        close_figure: close figure after display/save to save memory
        returnType: = ['figure','Image']
    - Potential:
      verbose=False, 
    - Output:
        True if no error occur
    Ref: https://github.com/marksgraham/OCT-Converter/blob/master/oct_converter/image_types/oct.py
    Ref (Colormap): https://matplotlib.org/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
    '''

    if isinstance(vol, (Path,str)):
      vol = image_io.load_vol(vol, **kwargs)
    if isinstance(overlay_vol, (Path,str)):
      overlay_vol = image_io.load_vol(overlay_vol, **kwargs)
    if isinstance(overlay_vol, torch.Tensor):
      overlay_vol = np.array(overlay_vol)

    # avoid X-display error
    if show is not True:
      import matplotlib
      matplotlib.use('Agg')

    # Get the volume value range
    if volClim == None:
      volClim = [vol.min(), vol.max()]
    # Get the label value range
    if (overlay_vol is not None) and (labelClim == None):
      labelClim = [overlay_vol.min(), overlay_vol.max()]

    ## preprocess raw val
    if permute_axes is not None:
      vol = np.transpose(vol, axes=permute_axes)
    elif transpose is True:
      vol = vol.transpose(1,0,2)
    if flipud is True:
      vol = np.flipud(vol)

    # if rescale is True:
    #   if rescaleTo == None:
    #     rescaleTo = [vol.min(), vol.max()]
    #   if threRange == None:
    #     ratio = 0.1
    #     eliminate = (rescaleTo[1] - rescaleTo[0])*ratio
    #     threRange = [rescaleTo[0]+eliminate, rescaleTo[1]-eliminate]
    #   rescaleVol(vol, threRange, rescaleTo)

    ## preprocess overlay vol
    if overlay_vol is not None:
      if permute_axes is not None:
        overlay_vol = np.transpose(overlay_vol, axes=permute_axes)
      elif transpose is True:
        overlay_vol = overlay_vol.transpose(1,0,2)
      if flipud is True:
        overlay_vol = np.flipud(overlay_vol)
      # removing background colors
      overlay_vol = removeBackground(overlay_vol)

    ## start plot visualization
    subfigs = rows * cols
    num_slices = vol.shape[-1] # last dimension
    x_size = rows * vol.shape[0]
    y_size = cols * vol.shape[1]
    ratio = y_size / x_size
    slices_indices = np.linspace(0,num_slices-1,subfigs).astype(np.int)

    # plot figure
    if isinstance(figsize, int):
      fig = plt.figure(figsize=(figsize*ratio,figsize))
    elif isinstance(figsize, tuple) and len(figsize) == 2:
      fig = plt.figure(figsize=figsize)
    else:
      print('figsize have to be int or tuple with size 2')

    for i in range(subfigs):
        slice_no = slices_indices[i]
        plt.subplot(rows, cols, i+1)

        # plot raw volume
        plt.imshow(vol[:,:,slice_no], cmap=volCmap,
                  vmin=volClim[0], vmax=volClim[1], aspect=aspect)
        plt.axis('off')
        if sliceNo==True:
          plt.title(f'{slice_no}')

        # overlay volume
        if overlay_vol is not None:
          if labelNorm is not None: # use labelNorm
            plt.imshow(overlay_vol[:,:,slice_no], 
                      cmap=labelCmap, norm=labelNorm,
                      alpha=alpha, aspect=aspect)
          else: # use vimn/vmax
            plt.imshow(overlay_vol[:,:,slice_no], cmap=labelCmap,
                      vmin=labelClim[0], vmax=labelClim[1],
                      alpha=alpha, aspect=aspect)

        # overlay surf
        if overlay_surf is not None:
          plt.plot(overlay_surf[:,slice_no,:])

    if fig_title is not None:
      # fig_title = f'Total number of slice: {num_slices}'
      plt.suptitle(fig_title)

    fig.tight_layout()

    # save or show
    if filepath is not None:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    if show is True:
        plt.show()

    # Close plot
    if close_figure is True:
      plt.close(plt.gcf())
      plt.close(fig)
      fig=None

    # Release memory
    vol=None

    if returnType == 'figure':
      return fig
    elif returnType == 'Image':
      return fig2img(fig, dpi=dpi, bbox='tight')
    else:
      return None

@delegates(vol_peek)
def vol_peek_seg_compare(vol:(np.array,torch.Tensor,Path,str), segGT:(np.array,torch.Tensor,Path,str), segAuto:(np.array,torch.Tensor,Path,str),**kwargs):
  '''compare GroundTruth with auto-segmentation results'''
  if isinstance(vol, (Path,str)):
    vol = image_io.load_vol(vol)
  if isinstance(segGT, (Path,str)):
    segGT = image_io.load_vol(segGT)
  if isinstance(segAuto, (Path,str)):
    segGT = image_io.load_vol(segAuto)

  #  
  vol_peek()
    

# Cell
#%% generate customized colormap for CT
def customize_cmap(tissueType='ABACS'):
  ''' generate customized colormap
  - BC: Body composition from CT
  # Ref: https://matplotlib.org/3.1.1/tutorials/colors/colorbar_only.html
  '''
  from med_deeplearning.data_io.faisal_io import faisal_abacs_getStandardTissueSettings as LUT
  from matplotlib.colors import ListedColormap
  import numpy as np
  if tissueType == "ABACS":
    # "abacs" colormap
    df = LUT()['dataFrame'].sort_values(by=['segmentation_label_value'])
    labels = np.array(df.segmentation_label_value)
    colors = np.array(tuple(zip(df.colour_r,df.colour_g,df.colour_b)))/255
    order = np.argsort(labels)
    # reorder labels/colors
    labels = labels[order]
    colors = colors[order,...]
    # bounds = np.array([labels[0]-1,*labels])+0.5 # labels # 
    bounds = np.array([*labels],labels[-1]+1) # labels # 
  elif tissueType == "BC":
    # Body composition only
    colors = np.array(((0,0,0),(255,0,0),(0,255,0),(0,0,255),(240,98,164),(255,255,0),(101,106,177),(0,255,255),(255,163,51)))/255
    bounds = [0,1,2,3,4,5,6,7,8,9] # [0-8] # [-1, 0, ...
  else: # if not pre-defined tissue Type
    cmap,norm = 'jet',None
    return cmap, norm
  
  cmap_customize = ListedColormap(colors=colors, name=tissueType)
  norm = mpl.colors.BoundaryNorm(bounds, cmap_customize.N)
  return cmap_customize, norm


# Cell
@delegates(vol_peek)
def quickcheck_generator(rawVol, segVol, qcPath,
                        sliceNo=True, dpi=300, 
                        quality=90, verbose=False,
                        rows_axial=10, cols_axial=10,
                        rows_sagittal=10, cols_sagittal=10,
                        rows_coronal=11, cols_coronal=9,
                        dest_axcode=None,
                        volCmap='gray', labelType='ABACS', 
                        figsize_axial=(15,15), figsize_sagittal=(15,20),
                        figsize_coronal=(15,25), 
                        permute_axial=[0,1,2], permute_sagittal=[2,0,1],
                        permute_coronal=[2,1,0],
                        flipud_axial=False, flipud_sagittal=True, 
                        flipud_coronal=True, thres=None, show=False,
                        rescale=False, threRange=None, rescaleTo=None, **kwargs):
  # load data
  if isinstance(rawVol, (str, Path)):
    if verbose > 0: print('loading raw volume ...')
    # set volume orientation
    if verbose > 1:
      print(f'reorient volumes to: {dest_axcode}')
    rawVol = image_io.load_vol(rawVol, dest_axcode=dest_axcode)
  if isinstance(segVol, (str, Path)):
    if verbose > 0: print('loading seg volume ...')
    # set volume orientation
    if verbose > 1:
      print(f'reorient volumes to: {dest_axcode}')
    segVol = image_io.load_vol(segVol, dest_axcode=dest_axcode)

  if thres is not None:
    rawVol = image_io.threshold(rawVol, thres,)
  # threshold rawVal
  if rescale == True:
    if threRange is None: threRange = thres
    rawVol = image_io.intensity_scale(rawVol, scaleFrom=threRange, scaleTo=rescaleTo)
    # rawVol = rescaleIntensity(rawVol, threRange=threRange, rescaleTo=rescaleTo)

  labelCmap,labelNorm = customize_cmap(tissueType=labelType)
  if labelNorm is None:
    labelClim = [segVol.min(), segVol.max()]
  else:
    labelClim = [labelNorm.boundaries[0], labelNorm.boundaries[-1]]
    
  if verbose > 2:
    print(f'labelNorm.boundaries={labelNorm.boundaries}')
    print(f'labelCmap.N={labelCmap.N}')
    print(f'labelClim={labelClim}')
    print(f'labelType={labelType}')
    
  # axial
  if verbose > 1: print('gerenating axial quickcheck ...')
  axial = vol_peek(rows=rows_axial, cols=cols_axial, sliceNo=sliceNo, dpi=dpi, vol=rawVol.transpose(permute_axial), overlay_vol=segVol.transpose(permute_axial), show=show, returnType='Image', flipud=flipud_axial, volCmap=volCmap, labelCmap=labelCmap, labelNorm=labelNorm, figsize=figsize_axial, labelClim=labelClim) #, **kwargs)

  # sagittal
  if verbose > 1: print('gerenating sagittal quickcheck ...')
  sagittal = vol_peek(rows=rows_sagittal, cols=cols_sagittal, sliceNo=sliceNo, dpi=dpi, vol=rawVol.transpose(permute_sagittal), overlay_vol=segVol.transpose(permute_sagittal), show=show, returnType='Image', flipud=flipud_sagittal, volCmap=volCmap, labelCmap=labelCmap, labelNorm=labelNorm, figsize=figsize_sagittal, labelClim=labelClim) #, **kwargs)

  # coronal
  if verbose > 1: print('generating coronal quickcheck ...')
  coronal = vol_peek(rows=rows_coronal, cols=cols_coronal, sliceNo=sliceNo, dpi=dpi, vol=rawVol.transpose(permute_coronal), overlay_vol=segVol.transpose(permute_coronal), show=show, returnType='Image', flipud=flipud_coronal, volCmap=volCmap, labelCmap=labelCmap, labelNorm=labelNorm, figsize=figsize_coronal, labelClim=labelClim) #, **kwargs)

  # combined into a singla pdf
  imglist = [axial,sagittal,coronal]
  imglist[0].save(qcPath, save_all=True, append_images=imglist[1:], quality=quality, optimize=False, resolution=dpi)

  # remove all variables to release memory
  rawVol=None
  segVol=None 
  axial,sagittal,coronal=None,None,None
  
  # close all figures
  plt.close('all')

  return imglist

# Cell
@delegates(quickcheck_generator)
def untar_quickcheck_generator(tarPath, qcDir, rawFname, segFname, qcName='quickcheck.pdf', sliceNo=False, dpi=300, verbose=False, rescale=False, threRange=None, rescaleTo=None, **kwargs):
  '''generate quickcheck from tar file
  -Input:
    - tempDir: directory to store temporary file (default to memory buffer - io.ByteIO())

  '''
  import os
  from pathlib import Path

  tarPath = Path(tarPath)
  qcDir = Path(qcDir)

  # extract raw file
  if not os.path.isfile(tarPath/rawFname):
    if verbose is True: print(f'untar raw volume: {rawFname}')
    rawPath = image_io.untar(tarPath, qcDir, fname=rawFname)
  # extract seg file
  if not os.path.isfile(tarPath/rawFname):
    if verbose is True: print(f'untar seg volume: {segFname}')
    segPath = image_io.untar(tarPath, qcDir, fname=segFname)
  # generate quickcheck
  quickcheck_generator(rawPath, segPath, qcDir/qcName, sliceNo=sliceNo, dpi=dpi, verbose=verbose, rescale=rescale, threRange=threRange, rescaleTo=rescaleTo, **kwargs)

# Cell
# Visualizing in 3D
# Ref: https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed