#!/bin/bash
#SBATCH --job-name=Run_07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --output=logs/Run_07-%j.out
#SBATCH --error=logs/Run_07-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/07_knn.py "$1" "$2"
