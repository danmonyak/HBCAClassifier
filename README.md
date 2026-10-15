# Reed_Explore

Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.). This is the reference that `Cell_Type_Assignment/02_SingleR_Reed_transfer.R` uses to give our cells their level-2 labels. The folder was made from `Template_Analysis`.

Run everything from inside this directory, because figures are written to a relative `figures/` path.

## Running

```
sbatch Run_01.sh [cellranger_run_name]
```

## 01_Analysis.R

- **Needs:** `hbca-global.h5ad`, in the folder set as `HBCA_DIR` in `local_paths.env`.
- **Produces:** `figures/<run>/01/LASP_subtypes_per_donor.png`, a stacked bar plot with one bar per donor (55 donors) and the five LASP subtypes stacked.

What it computes (printed to the log, not saved):

- The atlas's size and its cell counts per `level1` and `level2` label.
- `donor.level2.props`: a donors x `level2` subtypes table. Each value is the fraction of that donor's cells with that subtype, so every row sums to 1. `donor.level2.props.df` is the same table as a data frame, with `donor_id` as the first column. The log shows its size and first rows.
- `donor.lasp.props` (and `donor.lasp.props.df`): the same kind of table for the LASP subtypes only (`LASP1`–`LASP5`), as a fraction of each donor's LASP cells, so every row again sums to 1. The log shows the full table and each donor's number of LASP cells.

**Some donors have very few LASP cells** (the smallest has 14), so their proportions are noisy.

What it sets up:

- The standard header: this folder's `config.R`, `cellranger_run_name` (default `90`), `output_dir` (`storage/<run>/`) and `figures/<run>/01/`.
- Loads the atlas as `reed`, a SingleCellExperiment, the same way as `02_SingleR_Reed_transfer.R`. Its main `colData` columns are `level0`, `level1` and `level2` (the label hierarchy), `cell_type`, `donor_id`, `risk_status`, `brca_status` and `sample_type`. Row names are Ensembl IDs, like ours. Gene symbols are in `rowData(reed)$feature_name`.
- The analysis itself goes in the "Analysis" section, before `sessionInfo()`.

**The atlas is large**: 803,283 cells x 33,604 genes, 8.3 GB on disk.

## Other files

- `config.R`: settings for this folder (`figure_outdir`, point sizes, `rng.seed`). It sources the shared `RNAseq_SingleCell/config.R`.
- `Run_01.sh`: SLURM script for `01_Analysis.R`. It requests 100G and 1 day on `htc`, and runs in the `renv` environment. 100G matches `Cell_Type_Assignment/Run_02.sh`, which also loads the atlas.
- `output_files/`, `error_files/`: SLURM stdout and stderr.
