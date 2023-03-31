#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=20G

/home/susobhan/Data/susobhan/miniconda/envs/gwecc/bin/activate

echo $CONDA_PREFIX
date

python run_1psr_analysis.py 1psr_analysis.json
