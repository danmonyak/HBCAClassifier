#!/bin/bash
#SBATCH --job-name=Run_02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --partition=htc
#SBATCH --output=logs/Run_02-%j.out
#SBATCH --error=logs/Run_02-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate renv

Rscript scripts/02_marker_matrices.R
