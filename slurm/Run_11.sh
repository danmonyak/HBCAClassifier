#!/bin/bash
#SBATCH --job-name=Run_11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --output=logs/Run_11-%j.out
#SBATCH --error=logs/Run_11-%j.err
#SBATCH --time=04:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1
# So scripts/ can import the hbca_classifier package (jobs start in the submit folder, the repo root)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/11_transplant_thresholds.py "$1" "$2"
