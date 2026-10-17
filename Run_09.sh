#!/bin/bash
#SBATCH --job-name=Run_09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=htc
# One task per TRANSPLANT_GAMMAS value (network + baseline), plus one for KNN (level2_classifier_common.py)
#SBATCH --array=0-5
#SBATCH --output=output_files/Run_09-%A_%a.out
#SBATCH --error=error_files/Run_09-%A_%a.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 09_Transplant_Tier2.py "$1"
