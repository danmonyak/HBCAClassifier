#!/bin/bash
#SBATCH --job-name=Run_09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=htc
# One task per TRANSPLANT_GAMMAS value (network + baseline), plus one for KNN (hbca_classifier/settings.py)
#SBATCH --array=0-5
#SBATCH --output=logs/Run_09-%A_%a.out
#SBATCH --error=logs/Run_09-%A_%a.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/09_transplant.py
