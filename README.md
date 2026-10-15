# Reed_Explore

Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.). This is the reference that `Cell_Type_Assignment/02_SingleR_Reed_transfer.R` uses to give our cells their level-2 labels. The folder was made from `Template_Analysis`.

Run everything from inside this directory, because figures are written to a relative `figures/` path.

## Running

```
sbatch Run_01.sh [cellranger_run_name]
```

## 01_Analysis.R

- **Needs:** `hbca-global.h5ad`, in the folder set as `HBCA_DIR` in `local_paths.env`.
- **Produces:** nothing yet. It prints the atlas's size and its cell counts per `level1` and `level2` label, then `sessionInfo()`.

What it sets up:

- The standard header: this folder's `config.R`, `cellranger_run_name` (default `90`), `output_dir` (`storage/<run>/`) and `figures/<run>/01/`.
- Loads the atlas as `reed`, a SingleCellExperiment, the same way as `02_SingleR_Reed_transfer.R`. Its main `colData` columns are `level0`, `level1` and `level2` (the label hierarchy), `cell_type`, `donor_id`, `risk_status`, `brca_status` and `sample_type`. Row names are Ensembl IDs, like ours. Gene symbols are in `rowData(reed)$feature_name`.
- The analysis itself goes in the "Analysis" section, before `sessionInfo()`.

**The atlas is large**: 803,283 cells x 33,604 genes, 8.3 GB on disk.

## Other files

- `config.R`: settings for this folder (`figure_outdir`, point sizes, `rng.seed`). It sources the shared `RNAseq_SingleCell/config.R`.
- `Run_01.sh`: SLURM script for `01_Analysis.R`. It requests 100G and 1 day on `htc`, and runs in the `renv` environment. 100G matches `Cell_Type_Assignment/Run_02.sh`, which also loads the atlas.
- `output_files/`, `error_files/`: SLURM stdout and stderr.
