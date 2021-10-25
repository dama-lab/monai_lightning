
#%% import libraries

#%% Generalized Linear Models
# Ref: https://www.statsmodels.org/stable/glm.html
# Ref: https://aichamp.wordpress.com/2017/03/09/getting-p-values-from-glm-model-in-python/
# Ref: https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html
# Ref: https://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html
# Ref: https://datatofish.com/multiple-linear-regression-python/
# Ref: https://scipy-lectures.org/packages/statistics/auto_examples/plot_regression_3d.html
# Ref: https://scikit-learn.org/stable/modules/linear_model.html
# Ref: https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/10_multiple_regression.html
# Ref: [Statistics in Python](https://scipy-lectures.org/packages/statistics/index.html)

import pandas as pd, numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

#%% grap two group matrics from a dataframe
def grab_group_data(df, data_titles, group_title, ref_group):
  '''
  data_titles: list of structures
  group_title: 'GENOTYPE'
  ref_group:   'WT'
  '''
  raw_mat = df[data_titles].to_numpy().transpose() # ==.transpose([1,0])
  dataset_array = df[group_title]
  dataset_list = dataset_array.unique()
  ref_dataset = 'WT'
  ref_group = raw_mat[:,dataset_array == ref_dataset]
  remaining_group = raw_mat[:,dataset_array != ref_dataset]
  return ref_group, remaining_group


#%% rename dataframe title
def df_rename(df, rename_dict):
  ''' 
  rename_dict: sample: {"1/2Cb": "Cb"}
  '''
  df_renamed = df.rename(columns=rename_dict)
  return df_renamed

# %%
def glm_fit(df, y_name=None, x_names=None, formula=None, verbose=False):
  '''Multi-variable Linear Regression
  Input:
  - df: input data frame 
  - Model specification:
  [Option 1]: using formula (sample formular: 'Cb ~ SEX_MALE + GENOTYPE_WT')
  [Option 2]: using y_name + x_names (list of strings)
    - x: pandas dataframe / pandas (e.g. x_names=['SEX_MALE', 'GENOTYPE_WT'])
    - y: pandas dataframe / pandas
  Return:
    model
    - to get p-value: model.pvalues (model.pvalues['SEX[T.MALE]'])
  ================================
  #%% ---- statsmodel GLM formula ---- (preferred) smf.glm
  # [limitation] need to change variable name (allowed name of variable is restricted)
  # - 1) cannot start with number; 2) cannot contain special characters
  # advantage: donot need to categorify/dummify the "object"-type variables
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #%% ---- statsmodel GLM api ---- (sm.glm)
  # [Limitation]: have to manually:
  # - dummify categorical variables
  # - add constant (intercept) term
  #%% ---- staticmodel OLS api ----
  (sm.OLS): special case for sm.glm
  '''
  # define formula
  if formula is None:
    if (y_name is None) or (x_names is None):
      print("either `formula` or `y_name`+`x_names` need to be spefcified")
      return
    else: # create formula based on the `y_name`+`x_names`
      formula = f"{y_name} ~ {' + '.join(x_names)}"
  # define/train model
  model = smf.glm(formula=formula, data=df).fit()
  # print summary
  if verbose is True:
    print(model.summary())
  return model

# %% calculate the GLM p-value for all structures
def glm_fit_pvalues(df, y_list, x_names, x_test=None):
  '''generate GLM p-value array for all structures
  Inuput:
    - df: input pandas dataframe
    - struc_list: list of structures in the df title to be tested as y variable (dependent/exdoge)
    - x_names: list of independent (indog) variables (e.g. ['SEX', 'GENOTYPE', 'total_gm_vol']) 
               need to follow allowed variable pattern (i.e. not start with number, and no special character)
    - x_test: the con
  Return:
    - p_array: array of p-values
  '''
  p_array = np.empty(len(y_list))
  for i,struct in enumerate(y_list):
    df_glm = df.rename(columns={struct:"Y"})
    p_array[i] = glm_fit(df_glm, y_name="Y", x_names=x_names).pvalues.filter(regex=x_test)[0]
    del(df_glm)
  return p_array
