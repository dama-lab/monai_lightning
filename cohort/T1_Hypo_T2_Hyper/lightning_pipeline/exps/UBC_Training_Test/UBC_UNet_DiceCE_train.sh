#!/bin/bash
conda_init
conda activate fastai

echo ">>> Running UBC_UNet_DiceCE_train.py <<<"
echo ">>> python location: " $(which python)

python UBC_UNet_DiceCE_train.py
