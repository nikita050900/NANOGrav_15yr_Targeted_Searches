#!/bin/bash

#SBATCH --job-name=NGC3115_output
#SBATCH --output=NGC3115_QuickCW_run.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

which python

python runQuickMCMC.py

echo "Run complete."
