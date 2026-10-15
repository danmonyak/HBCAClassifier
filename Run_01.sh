#!/bin/bash
#SBATCH --job-name=Run_01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --partition=htc
#SBATCH --output=output_files/Run_01-%j.out
#SBATCH --error=error_files/Run_01-%j.err
#SBATCH --time=1-00:00:00

source $HOME/.bashrc
conda activate renv

Rscript 01_Analysis.R $1
