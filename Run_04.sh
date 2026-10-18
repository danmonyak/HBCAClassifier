#!/bin/bash
#SBATCH --job-name=Run_04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=htc
# One task per (fold, l1_ratio): tasks 0 to N_FOLDS x len(L1_RATIOS) - 1 (level2_classifier_common.py)
#SBATCH --array=0-19
#SBATCH --output=output_files/Run_04-%A_%a.out
#SBATCH --error=error_files/Run_04-%A_%a.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 04_Level2_Classifier_CV.py "$1"
