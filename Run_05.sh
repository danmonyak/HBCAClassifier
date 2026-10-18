#!/bin/bash
#SBATCH --job-name=Run_05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_05-%j.out
#SBATCH --error=error_files/Run_05-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 05_Level2_Classifier_Fit.py "$1" "$2"
