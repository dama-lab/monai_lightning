#%%
import numpy as np, torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# %%%%%%%%%%%%%%%%%%% QuickCheck %%%%%%%%%%%%%%%%%%%%%
def remove_background(labelVol, bgLabel=0, **kwargs):
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

# %% quickcheck   
def vol_peek(vol:tuple([np.array,torch.Tensor,Path,str]),
            rows=5, cols=5,
            filepath=None, fig_title=None, permute_axes=None, transpose=False, flipud=False,
            rescale=False, threRange=None, rescaleTo=None, overlay_vol:tuple([np.array, torch.Tensor,Path,str])=None, alpha=0.7,
            overlay_surf=None, linewidth=1, 
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
        overlay_surf: [layer,width,depth]
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
      mpl.use('Agg')

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
      overlay_vol = remove_background(overlay_vol)

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
          # [layer,width] ==transpose==> [width, layer]
          for layer in range(overlay_surf.shape[0]):
            plt.plot(overlay_surf[layer,:,slice_no], linewidth=linewidth)
          # alternatively, same as: 
          # plt.plot(overlay_surf[:,:,slice_no].transpose())


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

# =================================================================
#%% ==========   for visualizing the segmentation overlay   =======
# =================================================================
def get_label_bbox(lbl, dilate=0):
  '''find the min bbox containing the label mask'''
  import torch
  lbl = (lbl>0).type(torch.int)
  bbox_coors = torch.zeros(3,2).type(torch.int)
  axs = [[]]*3
  axs[0] = torch.amax(lbl,axis=(1,2))
  axs[1] = torch.amax(lbl,axis=(0,2))
  axs[2] = torch.amax(lbl,axis=(0,1))
  for i,ax in enumerate(axs):
    bbox_coors[i][0] = ax.argmax() - dilate
    bbox_coors[i][1] = len(ax) - ax.flip(dims=[0]).argmax() + dilate
  return bbox_coors
#%%
def crop_img_bbox(img, bb):
  '''crop image according to bbox coordinations
  bb: bounding box coordinations (shape = [3,2])
  '''
  return img[bb[0][0]:bb[0][1],bb[1][0]:bb[1][1],bb[2][0]:bb[2][1]]
#%%
def load_plot_batch(dataloader, crop=True, dilate=0, b=0):
  '''
  plot a batch from the dataloader
  '''
  # from med_deeplearning.HelperFunctions 
  import visualization as v
  val_batch = next(iter(dataloader()))
  img_b, lbl_b = val_batch['image'], val_batch['label']
  img, lbl = img_b[b,0,...], lbl_b[b,0,...]
  
  # crop the image based on the label bounding box
  if crop == True:
    # find the min bbox containing White-Matter-Leision
    bb = get_label_bbox(lbl, dilate)
    # plot batch data
    img_crop = crop_img_bbox(img,bb).detach()
    lbl_crop = crop_img_bbox(lbl,bb).detach()
  else:
    img_crop, lbl_crop = img, lbl

  fig = v.vol_peek(vol=img_crop, overlay_vol=lbl_crop)
  return fig, img, lbl
# %%