#!/bin/bash
#SBATCH --job-name=Run_08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_08-%j.out
#SBATCH --error=error_files/Run_08-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate base
export PYTHONUNBUFFERED=1

python 08_Level2_UMAP.py "$1" "$2"
