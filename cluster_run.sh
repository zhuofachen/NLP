#!/bin/bash -l

#$ -N res
#$ -P swangrp
#$ -o outputs/res
#$ -pe omp 4
#$ -l gpus=1 
#$ -l gpu_c=7
#$ -m ea
#$ -j y      
#$ -l h_rt=12:00:00
 
module load miniconda/4.7.5
conda activate FER
export PYTHONPATH=/projectnb/swangrp/zfchen/fer:$PYTHONPATH

python train.py network=resnet name=res save_freq=50
