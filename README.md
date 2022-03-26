# monai_lightning

## - Module library location:
`monai_pytorch_lightning`

## - yaml config file location:
`cohort/T1_Hypo_T2_Hyper/lightning_pipeline/exps/_wm_segs/`

## - main segmentation training function:
`cohort/T1_Hypo_T2_Hyper/lightning_pipeline/exps/seg3d.py`


----- requirement.txt -----
nibabel
matplotlib
seaborn

pytorch
pytorch-lightning
monai

nipype
scikit-image
scikit-learn
opencv

torchinfo

----- med-deeplearning -----
fastai (dangeous, will change torch to cpu version if not careful)
simpleitk
ipywidgets
albumentations