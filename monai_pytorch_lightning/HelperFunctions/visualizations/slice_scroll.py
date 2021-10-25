#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:22:43 2021

@author: dma73
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

#%%
# def previous_slice():
#     pass

# def next_slice():
#     pass

# def process_key(event):
#     if event.key == "left":
#         previous_slice()
#     elif event.key == "right":
#         next_slice()
        
#%%
# plt.rcParams['keymap.<command>'] = ['<key1>', '<key2>']

#%%
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def previous_slice(ax):
    ax.index = (ax.index - 1) % ax.volume.shape[0]  # wrap around using %
    # for i in range(len(ax.images)):
    ax.images[0].set_data(ax.volume[ax.index])
    if hasattr(ax,'label'):
      ax.images[1].set_data(ax.label[ax.index])
    
def next_slice(ax):
    ax.index = (ax.index + 1) % ax.volume.shape[0]
    # for i in range(len(ax.images)):
    ax.images[0].set_data(ax.volume[ax.index])
    if hasattr(ax,'label'):
      ax.images[1].set_data(ax.label[ax.index])
    
def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'left':
        previous_slice(ax)
    elif event.key == 'right':
        next_slice(ax)
    fig.canvas.draw()
    
def multi_slice_viewer(volume, label=None, alpha=0.5, vol_cmap='gray', label_cmap='jet'):
    
    # change to to channel-first
    volume = np.transpose(volume,[2,0,1])
    remove_keymap_conflicts({'left', 'right'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.alpha = alpha
    ax.vol_cmap = vol_cmap
    ax.label_cmap = label_cmap
    ax.imshow(ax.volume[ax.index],cmap=vol_cmap)
    if label is not None:
      label = np.transpose(label,[2,0,1])
      ax.label = label
      ax.imshow(ax.label[ax.index],alpha=alpha, cmap=label_cmap)
    fig.canvas.mpl_connect('key_press_event', process_key)
    return fig, ax
#%%
