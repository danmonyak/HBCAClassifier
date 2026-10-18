#!/bin/bash
#SBATCH --job-name=Run_06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_06-%j.out
#SBATCH --error=error_files/Run_06-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate torch
export PYTHONUNBUFFERED=1

python 06_Level2_NN.py "$1" "$2" "$3"
