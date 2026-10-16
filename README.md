# Reed_Explore

Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.). This is the reference that `Cell_Type_Assignment/02_SingleR_Reed_transfer.R` uses to give our cells their level-2 labels. The folder was made from `Template_Analysis`.

Run everything from inside this directory, because figures are written to a relative `figures/` path.

## Running

```
sbatch Run_01.sh [cellranger_run_name]   # 01_Analysis.R
sbatch Run_02.sh [cellranger_run_name]   # 02_Marker_Gene_Matrices.R
```

The two scripts don't depend on each other. The level2 classifier (`03`, `04`, `05`) needs `02`'s output (see below for how to run it).

## 01_Analysis.R

- **Needs:** `hbca-global.h5ad`, in the folder set as `HBCA_DIR` in `local_paths.env`.
- **Produces:** three stacked bar plots in `figures/<run>/01/`, each with one bar per donor (55 donors) and that cell type's subtypes stacked: `LASP_subtypes_per_donor.png` (`LASP1`–`LASP5`), `LHS_subtypes_per_donor.png` (`LHS1`–`LHS3`) and `BMYO_subtypes_per_donor.png` (`BMYO1`–`BMYO2`).

What it computes (printed to the log, not saved):

- The atlas's size and its cell counts per `level1` and `level2` label.
- `donor.level2.props`: a donors x `level2` subtypes table. Each value is the fraction of that donor's cells with that subtype, so every row sums to 1. `donor.level2.props.df` is the same table as a data frame, with `donor_id` as the first column. The log shows its size and first rows.
- `donor.lasp.props`, `donor.lhs.props` and `donor.bmyo.props` (each also as a `.df` data frame): the same kind of table for one cell type's subtypes only, as a fraction of each donor's cells of that type, so every row again sums to 1. They're made by `get_donor_subtype_props()`, which also saves the bar plot. The log shows each full table and each donor's cell count for that type.

**Some donors have very few cells of a type, so their proportions are noisy.** The smallest counts are 14 LASP, 26 LHS and 2 BMYO cells.

What it sets up:

- The standard header: this folder's `config.R`, `cellranger_run_name` (default `90`), `output_dir` (`storage/<run>/`) and `figures/<run>/01/`.
- Loads the atlas as `reed`, a SingleCellExperiment, the same way as `02_SingleR_Reed_transfer.R`. Its main `colData` columns are `level0`, `level1` and `level2` (the label hierarchy), `cell_type`, `donor_id`, `risk_status`, `brca_status` and `sample_type`. Row names are Ensembl IDs, like ours. Gene symbols are in `rowData(reed)$feature_name`.
- The analysis itself goes in the "Analysis" section, before `sessionInfo()`.

**The atlas is large**: 803,283 cells x 33,604 genes, 8.3 GB on disk.

## 02_Marker_Gene_Matrices.R

- **Needs:** `hbca-global.h5ad` (as above) and `data/HBCA_Reed_Top_Marker_Genes.xlsx`.
- **Produces**, all in `<STORAGE_DIR>/<run>/`:
  - `reed_celltype_marker_expr.rds` and `reed_state_marker_expr.rds`: the two matrices (`celltype.marker.expr` and `state.marker.expr`), genes x all atlas cells, with gene symbols as row names.
  - `reed_level1.rds` and `reed_level2.rds`: each atlas cell's `level1` and `level2` label.
  - The same data for Python: each matrix as `<name>.mtx.gz` (sparse Matrix Market), plus `<name>.genes.txt.gz` and `<name>.cells.txt.gz` (row and column names, in order), and `reed_cell_labels.csv.gz` (one row per cell: `cell`, `level1`, `level2`, in the matrices' cell order). Read them with `scipy.io.mmread` and `pandas.read_csv`. The script's comments show the exact lines.

What it does:

- Loads the atlas as `reed`, the same way as `01_Analysis.R`. **The values are log-normalized expression**: the atlas's `X`, the only matrix `read_h5ad` loads. The file also has raw counts in a separate `raw` slot, which isn't loaded.
- Reads the two sheets of the marker workbook, `celltype` and `state`. Each sheet has one column per cell type or state, with up to 100 marker gene symbols ranked from row 1. They become two named lists, `celltype.markers` and `state.markers` (cell type or state -> its genes, in rank order).
- Keeps the atlas genes whose symbol (`rowData(reed)$feature_name`) is one of the markers, and uses the symbols as the matrices' row names. The log lists the markers that aren't in the atlas. Checked against the files: 856 of 900 cell type markers and 2,968 of 3,155 cell state markers are in the atlas. Most of the missing ones are clone-based names like `AC093001.1`.

## The level2 classifier: 03, 04 and 05

Three Python steps that predict each atlas cell's epithelial cell state from its expression of the cell state marker genes. They run in the `torch` conda environment (see below), and share their settings, in `level2_classifier_common.py`. Run them in order, from inside this folder:

```
a=$(sbatch --parsable Run_03.sh [run] [cap])
b=$(sbatch --parsable --dependency=afterok:$a --kill-on-invalid-dep=yes Run_04.sh [run] [cap])
sbatch --dependency=afterok:$b --kill-on-invalid-dep=yes Run_05.sh [run] [cap]
```

`--dependency=afterok` starts each step only if the previous one succeeded. **Without `--kill-on-invalid-dep=yes`, a step whose predecessor failed stays in the queue forever as `DependencyNeverSatisfied`**, and you have to `scancel` it yourself.

Each takes two optional arguments: `cellranger_run_name` (default `90`) and `max_cells_per_class` (default: every cell). The cap keeps at most that many cells of each class, for a quicker run. **A capped run writes to its own folder, `reed_level2_classifier_cap<N>/`, and its figures to `figures/<run>/05/cap<N>/`,** so it never overwrites the full run.

The model:

- **Classes:** the 10 epithelial cell states (`LASP1`–`LASP5`, `LHS1`–`LHS3`, `BMYO1`–`BMYO2`). Every other `level2` label becomes `"Other"`. `"Other"` is by far the largest class (about 460,000 of 803,283 cells), so the results include balanced accuracy and per-class scores as well as accuracy.
- **Model:** multinomial logistic regression with a group elastic net penalty. Each gene's coefficients across all 11 classes form one group. The lasso part drops whole genes, and the ridge part spreads weight across correlated genes. The intercept isn't penalized.
- **Features:** the marker genes' log-normalized expression, scaled to unit variance on the training data only (not centred, which keeps the matrix sparse).
- **Tuning:** a stratified 70/30 train/test split, then 5-fold stratified cross-validation on the training set over `alpha` (10 values per `l1_ratio`) and `l1_ratio` (0.25, 0.5, 0.75, 1). Models are scored by held-out log-loss. The best pair is refitted on the whole training set and evaluated on the test set.

### 03_Level2_Classifier_Prep.py

- **Needs:** from `02`, in `<STORAGE_DIR>/<run>/`: `reed_state_marker_expr.mtx.gz`, `.genes.txt.gz`, `.cells.txt.gz` and `reed_cell_labels.csv.gz`.
- **Produces:** in `<STORAGE_DIR>/<run>/reed_level2_classifier/`: `X.npz` (cells x genes expression, unscaled) and `prep.npz` (the labels, the train/test split, each training cell's CV fold, and the alpha grid for each `l1_ratio`).

The alpha grids are set once from the whole training set, so every fold uses the same values and their results can be averaged. `Run_03.sh`: 4 CPUs, 64G.

### 04_Level2_Classifier_CV.py

- **Needs:** `X.npz` and `prep.npz` from `03`.
- **Produces:** `cv/fold<k>_l1ratio<r>.csv` in the same folder: one row per alpha, with the held-out log-loss, accuracy, balanced accuracy, genes kept and whether the solver converged.

`Run_04.sh` is a SLURM array of 20 tasks, one per fold and `l1_ratio`. Each fits the whole alpha path on 16 CPUs (48G). **If you change `N_FOLDS` or `L1_RATIOS`, change the `--array` line to match.**

### 05_Level2_Classifier_Fit.py

- **Needs:** `X.npz`, `prep.npz` and all 20 `cv/` files. It stops and lists them if any are missing.
- **Produces:** in the same folder:
  - `cv_summary.csv`: the mean CV results for every `l1_ratio` and `alpha`.
  - `coef.csv`: the fitted coefficients (the intercept, then one row per gene; one column per class, plus an `importance` column).
  - `test_report.csv`: precision, recall and F1 per class on the test set.
  - `model.npz`: `W`, `b`, the genes, classes and per-gene `scale`, to apply the model to other cells: probabilities = softmax((x / scale) @ W + b), with x the log-normalized expression of the same genes.
  - In `figures/<run>/05/`: `cv_curves.png` and `test_confusion_matrix.png`.

The log has the test accuracy, balanced accuracy and log-loss, and the top genes. `Run_05.sh`: 16 CPUs, 48G.

### The model code

- `multinomial_group_enet_torch.py` fits the model with PyTorch, by accelerated proximal gradient (FISTA). Each step is a few large matrix products, so it uses every CPU core the job has, and it would also run on a GPU (`device="cuda"`).
- `multinomial_group_lasso.py` is the same model for skglm. The scripts don't use it. It's kept to check the PyTorch solver. The two agreed on simulated data and on 300 real cells (probabilities within about 1e-7). **skglm is far slower:** on 20,000 cells, one alpha took up to 47 minutes with skglm, against 11 seconds with PyTorch on 16 CPUs. The model was also checked against scikit-learn: unpenalized and pure-ridge fits give the same probabilities to within 1e-7.

### The torch environment

The classifier steps need PyTorch, which is in its own conda environment, `torch` (about 7 GB):

```
conda create -y -n torch python=3.12 numpy scipy pandas scikit-learn h5py openpyxl matplotlib seaborn ipykernel pip
conda activate torch
PYTHONNOUSERSITE=1 python -m pip install torch skglm
```

**Use `python -m pip`, not `pip`.** Otherwise the cluster's system `pip` may be found first, and it installs into `~/.local` instead of the environment.

## Other files

- `config.R`: settings for this folder (`figure_outdir`, point sizes, `rng.seed`). It sources the shared `RNAseq_SingleCell/config.R`.
- `Run_02.sh`: SLURM script for `02_Marker_Gene_Matrices.R`, with the same resources as `Run_01.sh`.
- `level2_classifier_common.py`: the classifier's settings (classes, split, folds, penalty grid, solver tolerance) and shared helpers.
- `Run_03.sh`, `Run_04.sh`, `Run_05.sh`: SLURM scripts for the classifier steps (`torch` environment).
- `Run_01.sh`: SLURM script for `01_Analysis.R`. It requests 100G and 1 day on `htc`, and runs in the `renv` environment. 100G matches `Cell_Type_Assignment/Run_02.sh`, which also loads the atlas.
- `output_files/`, `error_files/`: SLURM stdout and stderr.
