import os,threading, pandas as pd, numpy as np
from glob import glob
from importlib import reload
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#%% === t-test
def cal_multi_ttest(raw_mat, dataset_array, ref_dataset,alpha=0.05):
  '''perform t-test across the whole array followed with FDR correction for multiple comparison
  - alpha: for FDR
  '''
  from scipy import stats
  from statsmodels.stats.multitest import fdrcorrection
  ref_mat = raw_mat[:,dataset_array==ref_dataset]
  alt_mat = raw_mat[:,dataset_array!=ref_dataset]
  # 2nd dim: across subjects (axis = 1)
  t_array, p_array = stats.ttest_ind(ref_mat, alt_mat,axis=1)
  # fdr correction
  fdrcorrection(p_array,alpha=alpha)
  return t_array, p_array

#%% === calculate the zscape ====  
def cal_zscore(raw_mat, dataset_array, dataset_list=None, ref_dataset=None):
  ''' calculate zscape
  raw_mat: [r * t] matrix, r = roi_list, t = target_list. exp:
    - t = 22 subjects (12 in reference, 10 in other group), 
    - r = 15 rois
  dataset_array: array of dataset name for each target
  dataset_list: list of dataset names [e.g. WT, TG]
  ref_dataset:  reference dataset ()
  '''
  # get the reference group
  if dataset_list is None:
    dataset_list = np.unique(dataset_array)
  if ref_dataset is None:
    ref_dataset = dataset_list[0] # e.g. 'NC' or 'WT'
  ref_group = raw_mat[:,dataset_array == ref_dataset]
  # calculate the ref group mean/std
  ref_mean = ref_group.mean(axis=1,keepdims=True)
  ref_std = ref_group.std(axis=1,keepdims=True)
  # calculate the zscape according to reference group mean/std
  zscores = (raw_mat - ref_mean)/ref_std
  # construct zscore_dict
  zscore_dict = {}
  zscore_dict['zscores'] = zscores
  zscore_dict['ref_group'] = ref_group
  zscore_dict['remaining_group'] = raw_mat[:,dataset_array != ref_dataset]
  return zscore_dict
#%% ==== calculate the wscore ===
def cal_wscore(raw_mat):
  return

#%% ==== plot zscape ====
def plot_zscape(zscores, dataset_array, dataset_list,threshold=0.5, vmin=-4, vmax=4, cmap='jet', p_array=None, ylabel=None, p_thres=0.05, ax=None):
  ''' plot zscape
  - zscores: [r * t] matrix, r=roi_list, t=target_list.   
    exp:
    - t = 22 subjects (12 in reference, 10 in other group), 
    - r = 15 roi
  - dataset_array: 
  - p_thres: thres of p-value to be significant (will be displayed as red color on the right axis)
  '''
  # zscape threshold
  zscores_thres = zscores.copy()
  zscores_thres[abs(zscores)<threshold]=np.nan

  ###  get the target number in each dataset
  datasets_sizes = np.array([np.nan]*len(dataset_list))
  for d, dataset in enumerate(dataset_list):
    datasets_sizes[d] = (dataset == dataset_array).sum()

  ### get location of separation lines
  sep_lines = np.array([np.nan]*(len(dataset_list)+1))
  sep_lines[0] = 0
  sep = 0
  for d,dsize in enumerate(datasets_sizes):
    sep = sep + dsize
    sep_lines[d+1] = sep

  ### calculate the dataset_name_title_loc
  dataset_name_title_loc = sep_lines[:-1] + (sep_lines[1:]-sep_lines[:-1])/2
  
  # ==========plot zscape using sns=========
  if vmin==None or vmax==None:
    ax = sns.heatmap(zscores_thres, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(zscores_thres, vmin=vmin,vmax=vmax, cmap=cmap, ax=ax)
  
  # plot dataset separate lines
  for sep_shift in sep_lines:
    ax.plot([sep_shift,sep_shift],[0,zscores_thres.shape[0]],color='k')

  ax_bottom = ax # plt.gca()
  ## plot y axis label
  # setup ytick
  if ylabel is not None:
    ax_bottom.set_yticklabels(ylabel, rotation=0)
    ax_bottom.set_yticks(np.arange(len(ylabel))+0.5)
  # add top axis for dataset title
  # Ref: https://www.kite.com/python/answers/how-to-add-a-second-x-axis-in-a-matplotlib-graph-in-python
  ax_top = ax_bottom.twiny()
  ax_top.set_xticks(dataset_name_title_loc/zscores_thres.shape[1])
  # add dataset name at top title
  ax_top.set_xticklabels(dataset_list)
  
  # add p-value on the right
  ax_right = None
  if p_array is not None:
    ax_right = ax_bottom.twinx()
    # align right yaxis
    ax_right.set_ylim(ax_bottom.get_ylim())
    ax_right.set_yticks(ax_bottom.get_yticks())
    # assign p-value to the right ytick label
    p_ticklabel = [f"{p:.3f}" for p in p_array]
    ax_right.set_yticklabels(p_ticklabel)
    # red-color right ytick label if p-value is significant
    [t.set_color('red') for i,t in enumerate(ax_right.yaxis.get_ticklabels()) if p_array[i] < p_thres]

  return ax, ax_bottom, ax_top, ax_right

def gen_zscape(raw_mat, dataset_array, dataset_list=None, ref_dataset=None, ylabel=None, threshold=0.5, vmin=-4, vmax=4, cmap='jet', cal_pvalue=True, p_thres=0.05, ax=None):
  # calculate zscore
  zscore_dict = cal_zscore(raw_mat, dataset_array, dataset_list=dataset_list, ref_dataset=ref_dataset)
  zscores = zscore_dict['zscores']
  # calculate p value
  if cal_pvalue is True:
    t_array, p_array = cal_multi_ttest(raw_mat, dataset_array, ref_dataset,alpha=0.05)
  else:
    p_array = None
  # plot zscape
  ax, ax_bottom, ax_top, ax_right = plot_zscape(zscores, dataset_array, dataset_list, p_array=p_array, p_thres=p_thres, ylabel=ylabel, threshold=threshold, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax)
  return ax, ax_bottom, ax_top, ax_right

#%% === Zscape class ===
class Zscape():
    def __init__(self, raw_mat, dataset_array, dataset_list,
                ref_dataset=None, threshold=0.5):
        self.raw_mat = raw_mat
        self.dataset_array = dataset_array
        self.dataset_list = dataset_list
        self.ref_dataset = ref_dataset
        self.threshold = threshold

    def cal_zsore(self):
        zscores, zscape_thres = cal_zscape(self.raw_mat, self.dataset_array, self.dataset_list, ref_dataset=self.ref_dataset, threshold=self.threshold)
        self.zscores, self.zscape_thres = zscores, zscape_thres

    def gen_zscape(self):
      return
