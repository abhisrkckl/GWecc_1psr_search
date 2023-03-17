#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=32G

/home/susobhan/Data/susobhan/miniconda/envs/gwecc/bin/activate

echo $CONDA_PREFIX
date

python run_analysis_j1909m3744.py
