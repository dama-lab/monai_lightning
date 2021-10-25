import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def recolor_boxplot(box,c):
  for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(box[item], color=c);
  plt.setp(box["boxes"], facecolor=c);
  plt.setp(box["fliers"], markeredgecolor=c);

def add_pvalues_to_right(ax_bottom, p_array, p_thres=0.05, fontsize=20):
  # add p-value on the right
  ax_right = None
  if p_array is not None:
    ax_right = ax_bottom.twinx()
    # align right yaxis
    ax_right.set_ylim(ax_bottom.get_ylim())
    ax_right.set_yticks(ax_bottom.get_yticks())
    # assign p-value to the right ytick label
    p_ticklabel = [f"{p:.3f}" for p in p_array]
    ax_right.set_yticklabels(p_ticklabel, fontsize=fontsize)
    # red-color right ytick label if p-value is significant
    [t.set_color('red') for i,t in enumerate(ax_right.yaxis.get_ticklabels()) if p_array[i] < p_thres]
  return ax_right

def figplot_2groups(ref_group, remaining_group, title=None, yaxislabels=None, group_list=None, plot_pvalue=True, p_array=None, p_thres=0.05, fontsize=20, ax=None, legend_loc="lower right", figpath=None, figtype="boxplot"):
  '''boxplot of two groups
  figtype: "boxplot"/"violinplot"
  '''
  # ref: https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color

  struct_no = ref_group.shape[0]
  # get the current axis
  if ax is None:
    fig, ax = plt.subplots(figsize=(10,10))

  if figtype == "boxplot":
    box1 = plt.boxplot(ref_group.transpose(), widths=0.4, positions=np.array(range(struct_no)), vert=False, patch_artist=True)
    box2 = plt.boxplot(remaining_group.transpose(), widths=0.4, positions=np.array(range(struct_no))+0.5, vert=False, patch_artist=True)
    # set color
    recolor_boxplot(box1,'blue')
    recolor_boxplot(box2,'red')
  elif figtype == "violinplot":
    v1 = plt.violinplot(ref_group.transpose(), vert=False,positions=np.array(range(len(ref_group))), widths=0.3)
    v2 = plt.violinplot(remaining_group.transpose(),vert=False,positions=np.array(range(len(ref_group)))+0.4, widths=0.3)
  

  # reverse yaxis order # https://stackoverflow.com/questions/2051744/reverse-y-axis-in-pyplot
  plt.gca().invert_yaxis()
  # to move xaxis tick to top:
  # ax.xaxis.tick_top()

  # add yticklabels
  ax.set_yticks(np.array(range(ref_group.shape[0]))+0.25);
  if yaxislabels is None:
    ax.set_yticklabels(['']*len(ax.get_yticks()), fontsize=fontsize);
  else:
    ax.set_yticklabels(yaxislabels,rotation='horizontal', fontsize=fontsize);

  #%% add legend (ref: https://stackoverflow.com/questions/47528955/adding-a-legend-to-a-matplotlib-boxplot-with-multiple-plots-on-same-axes/65237307)
  if group_list is not None:
    if figtype == "boxplot":
      legend_elements = [box1["boxes"][0],box2["boxes"][0]]
    elif figtype == "violinplot":
      legend_elements = [v1["cbars"],v2["cbars"]]
  ax.legend(legend_elements, group_list, fontsize=fontsize, loc=legend_loc);

  # add p_value on the right
  if p_array is not None:
    ax_right = add_pvalues_to_right(ax, p_array, p_thres=p_thres, fontsize=fontsize)

  # add title
  if title is not None:
    ax.set_title(title,fontsize=fontsize)

  if figpath is not None:
    fig = plt.gcf()
    fig.savefig(figpath)
    fig.show('off')
    plt.close()
    plt.clf()
  else:
    return ax
