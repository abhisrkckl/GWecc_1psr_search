#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --job-name=gwecc-search

module load pkgsrc/2022Q2

/home/susobhan/Data/susobhan/miniconda/envs/gwecc/bin/activate

PYTHON=$CONDA_PREFIX/bin/python

echo Script executed by `whoami` at `date`
echo The conda environment is $CONDA_PREFIX
$PYTHON --version
mpichversion | head -n 1
$PYTHON -c 'import numpy as np; print("numpy version", np.__version__)'
$PYTHON -c 'import mpi4py as mpi; print("mpi4py version", mpi.__version__)'

$PYTHON run_1psr_analysis.py 1psr_analysis.json
