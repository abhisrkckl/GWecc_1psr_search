#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --job-name=gwecc-search

module load pkgsrc/2022Q2

/home/susobhan/Data/susobhan/miniconda/envs/gwecc1/bin/activate

PYTHON=$CONDA_PREFIX/bin/python
JULIA=$CONDA_PREFIX/bin/julia

echo Script executed by `whoami` at `date`
$PYTHON --version
$CONDA_EXE --version
echo The conda environment is $CONDA_PREFIX
$PYTHON -c 'import numpy as np; print("numpy version", np.__version__)'
$PYTHON -c 'import scipy as sp; print("scipy version", sp.__version__)'
$PYTHON -c 'import astropy as ap; print("astropy version", ap.__version__)'
$PYTHON -c 'import matplotlib as mpl; print("matplotlib version", mpl.__version__)'
$PYTHON -c 'import corner; print("corner version", corner.__version__)'
$PYTHON -c 'import pint; print("PINT version", pint.__version__)'
echo tempo2 version is `tempo2 -v`
$PYTHON -c 'import libstempo; print("libstempo version", libstempo.__version__)'
$PYTHON -c 'import enterprise; print("ENTERPRISE version", enterprise.__version__)'
$PYTHON -c 'import enterprise_extensions; print("enterprise_extensions version", enterprise_extensions.__version__)'
$PYTHON -c 'import PTMCMCSampler; print("PTMCMCSampler version", PTMCMCSampler.__version__)' | grep version
$JULIA --version
$PYTHON -c 'import enterprise_gwecc as gwecc; print("GWecc version", gwecc.__version__)'
# mpichversion | head -n 1
# $PYTHON -c 'import mpi4py as mpi; print("mpi4py version", mpi.__version__)'

echo

$PYTHON run_1psr_analysis.py 1psr_analysis.json
