#!/bin/bash
#SBATCH --job-name=Run_07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_07-%j.out
#SBATCH --error=error_files/Run_07-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 07_Level2_KNN.py "$1" "$2" "$3"
