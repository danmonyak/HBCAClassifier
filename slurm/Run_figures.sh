#!/bin/bash
#SBATCH --job-name=Run_figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=htc
#SBATCH --output=logs/Run_figures-%j.out
#SBATCH --error=logs/Run_figures-%j.err
#SBATCH --time=01:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/figures.py "$1"
