#!/bin/bash
#SBATCH --job-name=Run_05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=htc
#SBATCH --output=logs/Run_05-%j.out
#SBATCH --error=logs/Run_05-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/05_fit.py "$1" "$2"
