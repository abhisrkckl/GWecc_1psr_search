#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --job-name=gwecc-search

# module load pkgsrc/2022Q2

/home/susobhan/Data/susobhan/miniconda/envs/gwecc/bin/activate

PYTHON=$CONDA_PREFIX/bin/python
JULIA=$PYTHON_JULIACALL_BINDIR/julia

source print_info.sh
mpichversion | head -n 1
$PYTHON -c 'from mpi4py import MPI, __version__ as mpi_ver; print("mpi4py version", mpi_ver)'

echo

mpirun -np 4 $PYTHON run_1psr_analysis.py fix-wn_vary-rn_vary-ecw.json
