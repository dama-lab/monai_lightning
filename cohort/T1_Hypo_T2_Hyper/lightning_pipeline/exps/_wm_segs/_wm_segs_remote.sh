#!/bin/bash

source /home/dma73/miniconda3/bin/activate
# which conda python
conda activate monai_conda

python seg_training.py \
      -y "_wm_segs_remote.yaml" \
      -m "train"
      