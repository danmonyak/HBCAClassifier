#!/bin/bash
#SBATCH --job-name=Run_03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_03-%j.out
#SBATCH --error=error_files/Run_03-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 03_Level2_Classifier_Prep.py "$1" "$2"
