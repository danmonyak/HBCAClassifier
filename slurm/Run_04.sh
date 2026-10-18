#!/bin/bash
#SBATCH --job-name=Run_04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=htc
# One task per (fold, l1_ratio): tasks 0 to N_FOLDS x len(L1_RATIOS) - 1 (hbca_classifier/settings.py)
#SBATCH --array=0-19
#SBATCH --output=logs/Run_04-%A_%a.out
#SBATCH --error=logs/Run_04-%A_%a.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/04_cv.py "$1"
