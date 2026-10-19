# HBCAClassifier

*Software for feature selection and classification of breast epithelial cell states with scRNA-seq data*

[![DOI](https://zenodo.org/badge/1424663432.svg)](https://doi.org/10.5281/zenodo.23457813)

Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.), and a classifier that predicts its epithelial cell states (LASP1–5, LHS1–3, BMYO1–2) from a small set of marker genes. The goal is a set of about 20–40 genes that works as a predictor.

This started as the `RNAseq_SingleCell/Reed_Explore` folder of `tamoxifen-scRNA-seq`, where the HBCA is the reference for level-2 label transfer (`Cell_Type_Assignment/02_SingleR_Reed_transfer.R`). It moved here with its git history. **The copy in `tamoxifen-scRNA-seq` is no longer updated:** it still uses that repo's config and the `90` run name, and its outputs are in that repo's `storage/90/`.

Run everything from the repo root, because figures, logs and `data/` are relative paths.

## Layout

| Folder | What |
| --- | --- |
| `scripts/` | The steps, numbered in run order: `01_atlas_exploration.R`, `02_marker_matrices.R`, then the classifier, `03_prep.py` to `11_transplant_thresholds.py`; and `figures.py`, which redraws a run's test figures. |
| `slurm/` | One SLURM script per step, `Run_01.sh` to `Run_11.sh`, and `Run_figures.sh`. Submit them from the repo root, e.g. `sbatch slurm/Run_05.sh 20000 40`. |
| `hbca_classifier/` | The Python code the steps share: `config.py`, `settings.py`, `group_enet.py`, `tier2_models.py` and `plots.py` (see "Other files"). |
| `dev/` | Code the steps don't use: `multinomial_group_lasso.py`, the skglm version of the model, kept to cross-check the PyTorch solver. |
| `logs/` | SLURM stdout and stderr. |
| `data/` | The marker gene workbook (not in git). |
| `figures/` | Figures, in `figures/<step>[/cap<N>]/` (not in git). |

The Python steps import `hbca_classifier`, so the repo root must be on `PYTHONPATH`. The `slurm/` scripts set it. **To run a step outside SLURM, set it yourself,** e.g. `PYTHONPATH=$PWD python scripts/03_prep.py 300` from the repo root.

## Setup

1. **Paths:** copy `local_paths.env.example` to `local_paths.env` and fill in `STORAGE_DIR` (where every data object is saved and loaded) and `HBCA_DIR` (the folder with `hbca-global.h5ad`). `local_paths.env` is gitignored.
2. **Marker gene workbook:** put `HBCA_Reed_Top_Marker_Genes.xlsx` in `data/`. **It's kept out of git on purpose** (`data/*.xlsx` is gitignored).
3. **Conda environments:**
   - `renv` (R 4.5) for `01` and `02`: SingleCellExperiment, anndataR, readxl, dplyr, tibble, Matrix, ggplot2, tidySingleCellExperiment, patchwork, scuttle and scater.
   - `torch` for `03`–`07`, `09` and `10` (see "The torch environment" below).
   - The base conda environment for `08`, because it has umap-learn.

Every script reads and writes its data objects directly in `STORAGE_DIR`. There's no run-name level, unlike `tamoxifen-scRNA-seq`'s `storage/<run>/`.

## Running

```
sbatch slurm/Run_01.sh   # 01_atlas_exploration.R
sbatch slurm/Run_02.sh   # 02_marker_matrices.R
```

The two scripts don't depend on each other. The level2 classifier (`03`, `04`, `05`) needs `02`'s output (see below for how to run it).

## 01_atlas_exploration.R

- **Needs:** `hbca-global.h5ad`, in the folder set as `HBCA_DIR` in `local_paths.env`.
- **Produces:** three stacked bar plots in `figures/01/`, each with one bar per donor (55 donors) and that cell type's subtypes stacked: `LASP_subtypes_per_donor.png` (`LASP1`–`LASP5`), `LHS_subtypes_per_donor.png` (`LHS1`–`LHS3`) and `BMYO_subtypes_per_donor.png` (`BMYO1`–`BMYO2`).

What it computes (printed to the log, not saved):

- The atlas's size and its cell counts per `level1` and `level2` label.
- `donor.level2.props`: a donors x `level2` subtypes table. Each value is the fraction of that donor's cells with that subtype, so every row sums to 1. `donor.level2.props.df` is the same table as a data frame, with `donor_id` as the first column. The log shows its size and first rows.
- `donor.lasp.props`, `donor.lhs.props` and `donor.bmyo.props` (each also as a `.df` data frame): the same kind of table for one cell type's subtypes only, as a fraction of each donor's cells of that type, so every row again sums to 1. They're made by `get_donor_subtype_props()`, which also saves the bar plot. The log shows each full table and each donor's cell count for that type.

**Some donors have very few cells of a type, so their proportions are noisy.** The smallest counts are 14 LASP, 26 LHS and 2 BMYO cells.

What it sets up:

- The header: `config.R`, `output_dir` (`STORAGE_DIR`) and `figures/01/`.
- Loads the atlas as `reed`, a SingleCellExperiment, with `anndataR::read_h5ad`. Its main `colData` columns are `level0`, `level1` and `level2` (the label hierarchy), `cell_type`, `donor_id`, `risk_status`, `brca_status` and `sample_type`. Row names are Ensembl IDs. Gene symbols are in `rowData(reed)$feature_name`.
- The analysis itself goes in the "Analysis" section, before `sessionInfo()`.

**The atlas is large**: 803,283 cells x 33,604 genes, 8.3 GB on disk.

## 02_marker_matrices.R

- **Needs:** `hbca-global.h5ad` (as above) and `data/HBCA_Reed_Top_Marker_Genes.xlsx`.
- **Produces**, all in `<STORAGE_DIR>/`:
  - `reed_celltype_marker_expr.rds` and `reed_state_marker_expr.rds`: the two matrices (`celltype.marker.expr` and `state.marker.expr`), genes x all atlas cells, with gene symbols as row names.
  - `reed_level1.rds` and `reed_level2.rds`: each atlas cell's `level1` and `level2` label.
  - The same data for Python: each matrix as `<name>.mtx.gz` (sparse Matrix Market), plus `<name>.genes.txt.gz` and `<name>.cells.txt.gz` (row and column names, in order), and `reed_cell_labels.csv.gz` (one row per cell: `cell`, `level1`, `level2`, in the matrices' cell order). Read them with `scipy.io.mmread` and `pandas.read_csv`. The script's comments show the exact lines.

What it does:

- Loads the atlas as `reed`, the same way as `01_atlas_exploration.R`. **The values are log-normalized expression**: the atlas's `X`, the only matrix `read_h5ad` loads. The file also has raw counts in a separate `raw` slot, which isn't loaded.
- Reads the two sheets of the marker workbook, `celltype` and `state`. Each sheet has one column per cell type or state, with up to 100 marker gene symbols ranked from row 1. They become two named lists, `celltype.markers` and `state.markers` (cell type or state -> its genes, in rank order).
- Keeps the atlas genes whose symbol (`rowData(reed)$feature_name`) is one of the markers, and uses the symbols as the matrices' row names. The log lists the markers that aren't in the atlas. Checked against the files: 856 of 900 cell type markers and 2,968 of 3,155 cell state markers are in the atlas. Most of the missing ones are clone-based names like `AC093001.1`.

## The level2 classifier: 03 to 11

Three Python steps (`03`–`05`) that predict each atlas cell's epithelial cell state from its expression of the cell state marker genes. `06` (a neural network) and `07` (weighted KNN) then predict the class from only the genes that `05` selected. We call `05`'s model the tier-1 model (it chooses the genes) and `06`/`07` the tier-2 models. `08` plots a UMAP of all cells on the selected genes. `09` and `10` (TRANSPLANT) retrain the tier-2 models on the full data with a capped run's genes, and `11` tries other decision rules on the TRANSPLANT networks. They run in the `torch` conda environment (`08` in base; see Setup), and share their settings, in `hbca_classifier/settings.py`. Every figure of one model's test results has its test balanced accuracy and macro F1 in the title (`score_text` in `hbca_classifier/settings.py`). Cells left unclassified in `11` count as misses. **`05`, `06` and `07` save each model's test predictions, and draw its test figures from them,** so the figures can be redrawn without refitting (see "Test figures" below). Run `03`–`05` in order, from the repo root:

```
a=$(sbatch --parsable slurm/Run_03.sh [cap])
b=$(sbatch --parsable --dependency=afterok:$a --kill-on-invalid-dep=yes slurm/Run_04.sh [cap])
sbatch --dependency=afterok:$b --kill-on-invalid-dep=yes slurm/Run_05.sh [cap]
```

`--dependency=afterok` starts each step only if the previous one succeeded. **Without `--kill-on-invalid-dep=yes`, a step whose predecessor failed stays in the queue forever as `DependencyNeverSatisfied`**, and you have to `scancel` it yourself.

Each takes one optional argument, `max_cells_per_class` (default: every cell). The cap keeps at most that many cells of each class, for a quicker run. **A capped run writes to its own folder, `reed_level2_classifier_cap<N>/`, and its figures to `figures/05/cap<N>/`,** so it never overwrites the full run.

The model:

- **Classes:** the 10 epithelial cell states (`LASP1`–`LASP5`, `LHS1`–`LHS3`, `BMYO1`–`BMYO2`). Every other `level2` label becomes `"Other"`. `"Other"` is by far the largest class (about 460,000 of 803,283 cells), so the results include balanced accuracy and per-class scores as well as accuracy.
- **Model:** multinomial logistic regression with a group elastic net penalty. Each gene's coefficients across all 11 classes form one group. The lasso part drops whole genes, and the ridge part spreads weight across correlated genes. The intercept isn't penalized.
- **Class weights:** every cell is weighted by the inverse frequency of its class, so each class counts equally. This applies in the fits, in alpha_max and in the CV log-loss. **Without weights, the genes that only help rare classes are the first to be dropped:** in an unweighted run with 2,000 cells per class, LASP5 (209 cells in the whole atlas) lost its markers at about 30 genes, and its test F1 fell to 0.36.
- **Features:** the marker genes' log-normalized expression, scaled to unit variance on the training data only (not centred, which keeps the matrix sparse). See "Gene scaling" below for the two ways to compute the variance.
- **Tuning:** a train/test split by donor (about 70/30), then 5-fold cross-validation on the training set, also by donor, over `alpha` and `l1_ratio` (0.25, 0.5, 0.75, 1). For each `l1_ratio`, there are 20 alphas, evenly spaced on a linear scale from just below alpha_max (the smallest alpha that drops every gene) down to 0.01. So most of them are near the top, where models keep few genes. The aim is a model with few genes. Models are scored by held-out log-loss (class-weighted). The best pair is refitted on the whole training set and evaluated on the test set.

### Gene scaling

Tier 1 (`03`–`05`) divides each gene by its standard deviation on the training cells. **The cells are weighted by inverse class frequency, as in the loss, so every class counts equally** (`get_scaling_weights` in `hbca_classifier/settings.py`).

**Why:** in the full data, 57% of cells are "Other". If every cell counted equally, a gene expressed only in a small population would have a tiny standard deviation, so it would be scaled up a lot and become cheap for the lasso to keep. The first full-data run did that, and its 40-gene model picked genes like ASPM, AURKB, OXTR and ROPN1. With class-weighted scaling, the same run picked 38 broad marker genes, and its network's test balanced accuracy rose from 0.750 to 0.867 (both on a cell-level split; see "Earlier results"). Every-cell scaling was removed after that.

### 03_prep.py

- **Needs:** from `02`, in `<STORAGE_DIR>/`: `reed_state_marker_expr.mtx.gz`, `.genes.txt.gz`, `.cells.txt.gz` and `reed_cell_labels.csv.gz`. Also `hbca-global.h5ad` (in `HBCA_DIR`), for each cell's donor.
- **Produces:** in `<STORAGE_DIR>/reed_level2_classifier/`: `X.npz` (cells x genes expression, unscaled) and `prep.npz` (the labels, each cell's donor, the train/test split, each training cell's CV fold, and the alpha grid for each `l1_ratio`).

**Every split is by donor, not by cell,** so a model is always scored on donors it never saw. (Until 2026-10-19 the splits were by cell; see "Earlier results".) The atlas has 55 donors.

- **Train/test:** `donor_split` (in `hbca_classifier/settings.py`) holds out about 30% of the donors (`TEST_SIZE`). Some states come mostly from a few donors (one donor has 38% of the LASP5 cells), so a random donor set can leave a state badly over- or under-represented in the test set. It tries 2,000 random donor sets (`DONOR_SPLIT_TRIES`, seeded) and keeps the one whose per-state test shares are closest to 30%. The log lists the test donors and each state's test share.
- **CV folds:** scikit-learn's `StratifiedGroupKFold` over the training cells, grouped by donor, so all of a donor's cells are in one fold. The log shows each fold's donors and per-state shares.
- **Donors:** read from `obs/donor_id` in `hbca-global.h5ad` with `h5py`, matched to the cells by name.

The alpha grids are set once from the whole training set, so every fold uses the same values and their results can be averaged. alpha_max depends on the data and the `l1_ratio` (about 0.25 / `l1_ratio` on the 300-per-class test), so each `l1_ratio` has its own grid. The 0.01 floor (`ALPHA_MIN`) is shared. The log lists every value. `slurm/Run_03.sh`: 4 CPUs, 64G.

### 04_cv.py

- **Needs:** `X.npz` and `prep.npz` from `03`.
- **Produces:** `cv/fold<k>_l1ratio<r>.csv` in the same folder: one row per alpha, with the held-out log-loss (class-weighted), accuracy, balanced accuracy, macro F1 (the mean of the per-class F1 scores, so every class counts equally), genes kept and whether the solver converged.

`slurm/Run_04.sh` is a SLURM array of 20 tasks, one per fold and `l1_ratio`. Each fits the whole alpha path on 16 CPUs (48G). **If you change `N_FOLDS` or `L1_RATIOS`, change the `--array` line to match.**

### 05_fit.py

- **Needs:** `X.npz`, `prep.npz` and all 20 `cv/` files. It stops and lists them if any are missing.
- **Produces:** in the same folder:
  - `cv_summary.csv`: the mean CV results for every `l1_ratio` and `alpha`.
  - `coef.csv`: the fitted coefficients (the intercept, then one row per gene; one column per class, plus an `importance` column).
  - `test_report.csv`: precision, recall and F1 per class on the test set.
  - `model.npz`: `W`, `b`, the genes, classes and per-gene `scale`, to apply the model to other cells: probabilities = softmax((x / scale) @ W + b), with x the log-normalized expression of the same genes.
  - `test_predictions_tier1.npz`: the test cells, their true class and the model's probabilities.
  - In `figures/05/`: `cv_curves.png` (mean CV log-loss, macro F1 and genes kept against alpha, one line per `l1_ratio`; a black star marks the chosen model), and the three test figures (see "Test figures"): `test_confusion_matrix.png`, `test_confusion_matrix_precision.png` and `test_precision_recall.png`.

The log has the test accuracy, balanced accuracy, macro F1 and log-loss, and the top genes. `slurm/Run_05.sh`: 16 CPUs, 48G.

**A smaller model:** `05` takes an optional second argument, `max_genes`. It then picks the best model (lowest CV log-loss) among those that keep at most that many genes, refits it and evaluates it. For example, on the full data with at most 200 genes:

```
sbatch slurm/Run_05.sh none 200
```

(`none` means no cell cap.) **The cap must match a run whose `03` and `04` have finished:** `none` needs the full-data CV, and a capped run needs the same cap (e.g. `sbatch slurm/Run_05.sh 300 50`). If the CV results are missing, `05` stops and lists them. If no model is small enough, `05` stops and says how many genes the smallest one keeps. This reuses the results of `03` and `04` (no need to rerun them), but it does refit the model. Every output except `cv_summary.csv` gets a `_max<N>genes` suffix, e.g. `test_confusion_matrix_max200genes.png`, so it doesn't overwrite the unconstrained model. The cap applies to the genes kept averaged over the CV folds, so the refitted model can keep a few more or fewer genes. Its count is in the confusion matrix's title.

### 06_network.py

A dense neural network that predicts the class from only the genes `05` selected. Run it after `05`, with the same arguments, usually with a gene limit:

```
sbatch slurm/Run_05.sh 300 40
sbatch slurm/Run_06.sh 300 40
```

- **Needs:** `X.npz` and `prep.npz` from `03`, and `model[_max<G>genes].npz` from `05`.
- **Produces:** in the classifier folder, all with the `_max<G>genes` suffix when a gene limit is given:
  - `nn.pt`: the network's weights, plus its genes, classes and input scaling (mean and SD per gene).
  - `nn_comparison.csv`: test log-loss (class-weighted), accuracy, balanced accuracy and macro F1, for the network and the baseline.
  - `nn_test_report.csv` and `baseline_test_report.csv`: per-class precision, recall and F1 on the test set.
  - `baseline.npz`: the baseline's coefficients and input scaling, to apply it to other cells.
  - `test_predictions_nn.npz` and `test_predictions_baseline.npz`: the test cells, their true class and each model's probabilities.
  - In `figures/06/` (or `.../06/cap<N>/`): `nn_training_curve.png` (training and validation loss per epoch), and the three test figures for each model, with an `nn_` or `baseline_` prefix (see "Test figures").

What it does:

- **Genes:** the nonzero rows of `05`'s model. They were chosen using only the training set, so the test set stays untouched.
- **Data:** the training set from `03`, minus the cells of about 10% of the training donors, held back for validation (`NN_VAL_SIZE`, chosen with `donor_split` as for the test set). Each gene is centred and scaled on the remaining training part.
- **Network:** dense layers of 64 and 32 units, with ReLU and 10% dropout, and one output per class. It's trained with Adam (learning rate 1e-3, batches of 512) on cross-entropy. Training stops once the validation loss hasn't improved for 20 epochs, and the best epoch's weights are kept.
- **Baseline:** an unpenalized multinomial logistic regression on the same genes and cells. **If the network doesn't beat it, the nonlinearity isn't adding anything.**
- **Class weights:** both models weight each class by its inverse frequency in the training data, so every class counts equally, as in balanced accuracy and macro F1. The selection model in `04`/`05` uses the same weights. Without weights, `"Other"` (57% of the full atlas) would dominate.

The network settings (`NN_HIDDEN`, `NN_DROPOUT`, `NN_LEARNING_RATE`, `NN_BATCH_SIZE`, `NN_MAX_EPOCHS`, `NN_PATIENCE`, `NN_VAL_SIZE`) are in `hbca_classifier/settings.py`.

**Other network shapes:** an optional third argument names another hidden-layer layout from `NN_ARCHITECTURES` in `hbca_classifier/settings.py`. Every output, the baseline's included, then gets an extra `_<name>` suffix, so the default network's results are kept. For example, `deep` has five hidden layers (64, 64, 32, 32, 16):

```
sbatch slurm/Run_06.sh 20000 40 deep
```

Tested with 40 genes, it was no better than the default 64 → 32 network: test balanced accuracy 0.835 vs 0.848 at 2,000 cells per class, and 0.843 vs 0.842 at 20,000. `slurm/Run_06.sh`: 8 CPUs, 32G.

### 07_knn.py

A second tier-2 model: weighted k-nearest neighbours on the genes `05` selected. Run it after `05`, with the same arguments. It doesn't need `06`, but if `06` has run, the comparison table includes the network and baseline:

```
sbatch slurm/Run_07.sh 20000 40
```

- **Needs:** `X.npz` and `prep.npz` from `03`, and `model[_max<G>genes].npz` from `05`. Optionally `nn_comparison[_max<G>genes].csv` from `06`.
- **Produces:** in the classifier folder, all with the `_max<G>genes` suffix when a gene limit is given:
  - `knn_tuning.csv`: validation balanced accuracy and macro F1 for every k, gamma and beta.
  - `knn_test_report.csv`: per-class precision, recall and F1 on the test set.
  - `knn_comparison.csv`: test accuracy, balanced accuracy and macro F1 for KNN, plus the network and baseline rows from `06`.
  - `test_predictions_knn.npz`: the test cells, their true class and KNN's vote shares.
  - In `figures/07/` (or `.../07/cap<N>/`): `knn_tuning.png` (validation balanced accuracy over k x gamma, one panel per beta), and the three test figures with a `knn_` prefix (see "Test figures").

What it does:

- **Cells:** the same as `06`. The training set minus the same validation donors is the reference. The validation donors' cells are used for tuning. The test set is used once.
- **Gene scaling:** each gene is centred and divided by SD^beta, using the reference cells. Beta = 1 is unit variance, 0.5 is Pareto scaling, and 0 is no scaling (each gene keeps its own spread, so high-variance genes count more in the distance).
- **Votes:** each of the k nearest reference cells (Euclidean distance) votes for its class with weight 1 / distance times (1 / its class's frequency in the reference)^gamma. Gamma = 0 is a plain distance-weighted vote; gamma = 1 fully corrects for class imbalance. Exact matches (distance 0) dominate the vote, as in scikit-learn.
- **Tuning:** k (5, 10, 25, 50, 100, 200, 400), gamma (0, 0.25, 0.5, 0.75, 1) and beta (0, 0.5, 1) are chosen together by **validation balanced accuracy**. For each beta, the neighbours are found once, at the largest k, so the whole k x gamma grid costs about the same as one k. The log notes if the best k is at the edge of the grid.
- **No log-loss for KNN:** its class scores are vote shares, which move in steps of about 1/k, so log-loss isn't a fair score.

The settings (`KNN_K`, `KNN_GAMMAS`, `KNN_BETAS`, `KNN_QUERY_BATCH`) are in `hbca_classifier/settings.py`. The neighbour search is brute force, in batches, with PyTorch on all the job's CPUs. `slurm/Run_07.sh`: 16 CPUs, 32G.

### 08_umap.py

One UMAP per tier-1 run, on only the genes that run's tier-1 model selected, coloured by cell state. It includes every cell in the 10 epithelial states (LASP1–5, LHS1–3, BMYO1–2), each state in its own colour, plus a random sample of 100,000 other cells (`UMAP_OTHER_SAMPLE`) in light grey, as `"Other"`, drawn underneath. It shows how well those genes alone separate the states. Run it after `05`, with the same arguments:

```
sbatch slurm/Run_08.sh 20000 40
```

- **Needs:** `model[_max<G>genes].npz` from that run's `05`, and `X.npz` and `prep.npz` from `03` run on the full data (no cap), which hold every cell.
- **Produces:** `umap[_max<G>genes].csv.gz` in that run's classifier folder (cell, state, UMAP1, UMAP2), and `figures/08/` (or `.../08/cap<N>/`): `umap[_max<G>genes].png`.

Every run's UMAP uses the same cells (the `"Other"` sample is seeded), so they differ only in their genes. Genes are centred and scaled to unit variance over the plotted cells. **UMAP itself isn't seeded,** so it runs on all the job's cores, and its layout differs slightly each time it's run. **It runs in the base conda environment**, because the `torch` environment doesn't have umap-learn. The UMAP settings (`UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST`, `UMAP_OTHER_SAMPLE`) are in `hbca_classifier/settings.py`. `slurm/Run_08.sh`: 8 CPUs, 64G.

### TRANSPLANT: 09_transplant.py and 10_transplant_summary.py

**`09`–`11` were built on the cell-level split and haven't been rerun since the split moved to donors.** Their inputs and outputs are in `<STORAGE_DIR>/old_cell_split/`. The class-weighted gene scaling made them unnecessary: the full-data run now picks good genes by itself.

Retrains the tier-2 models on the full data, on the genes that a capped run's tier-1 model chose. By default these are the 25 genes from the 20,000-per-class run with a 40-gene limit (`TRANSPLANT_SOURCE_CAP`, `TRANSPLANT_SOURCE_MAX_GENES`). The reason: on the full data, tier 1 chose a worse gene set (test balanced accuracy about 0.75 against 0.84). A likely cause is that inverse-frequency class weights become extreme there, about 2,200:1. The UMAPs (`08`) show the same thing.

```
a=$(sbatch --parsable slurm/Run_09.sh)
sbatch --dependency=afterok:$a --kill-on-invalid-dep=yes slurm/Run_10.sh
```

- **Needs:** `X.npz` and `prep.npz` from `03` on the full data, and `prep.npz` and `model_max<G>genes.npz` from the source run.
- **Produces:** in `<STORAGE_DIR>/reed_level2_transplant/`:
  - From `09`: `gamma<g>.npz` per network/baseline task (validation scores and test probabilities), `nn_gamma<g>.pt`, `nn_training_curve_gamma<g>.csv`, `knn.npz`, `knn_tuning.csv` and `test_cells.csv.gz`.
  - From `10`: `gamma_tuning.csv`, `transplant_comparison.csv` (test log-loss, accuracy, balanced accuracy and macro F1 for the three chosen models), and `nn_test_report.csv`, `baseline_test_report.csv` and `knn_test_report.csv`. In `figures/10/`: `gamma_tuning.png` and a test confusion matrix per model.

What it does:

- **Cells:** the full run's training set minus a stratified 10% validation part (as in `06`), for training and tuning. **The test set is the full run's test set minus every cell in the source run's training set,** because the source run's tier-1 model chose the genes using those cells.
- **Tempered class weights:** each class is weighted by (1 / its frequency)^gamma, with gamma in `TRANSPLANT_GAMMAS` (0, 0.25, 0.5, 0.75, 1). `slurm/Run_09.sh` is an array with one task per gamma, each training the network and the unpenalized logistic regression, plus one task for KNN, which tunes its own k, gamma and beta as in `07`. **If you change `TRANSPLANT_GAMMAS`, change the `--array` line to match.**
- **Choosing:** `10` picks each model's gamma by validation balanced accuracy, then scores the test set once.

The network training and KNN votes come from `hbca_classifier/tier2_models.py`, which copies the code of `06` and `07` as functions (`06` and `07` keep their own copies). `slurm/Run_09.sh`: 16 CPUs, 32G per task. `slurm/Run_10.sh`: 4 CPUs, 16G.

### 11_transplant_thresholds.py

Other ways to turn the TRANSPLANT networks' probabilities into calls. **The unweighted network (trained gamma 0) is the tier-2 model;** the others are kept for comparison. There's one network per trained gamma (0, 0.25, 0.5, 0.75, 1), and nothing is retrained: `11` reloads the networks `09` saved and recomputes their probabilities on `09`'s validation and test cells. Run it after `09`:

```
sbatch slurm/Run_11.sh [other_threshold] [state_threshold]    # defaults 0.6 and 0.8
```

- **Needs:** from `09`, in `<STORAGE_DIR>/reed_level2_transplant/`: `nn_gamma<g>.pt` and `gamma<g>.npz` for every gamma, and `test_cells.csv.gz`. Also `09`'s own inputs (the full-data `X.npz` and `prep.npz`, and the source run's `prep.npz` and `model_max<G>genes.npz`), to rebuild its cells.
- **Produces:** in `<STORAGE_DIR>/reed_level2_transplant/thresholds/`:
  - `posthoc_validation.csv`: validation balanced accuracy, macro F1 and accuracy for every trained gamma x `gamma_post`.
  - `posthoc_test.csv`: for each trained gamma, test scores with the plain top class and with its best `gamma_post`.
  - `threshold_report_other<a>_state<b>.csv`: for each trained gamma, split and class: true cells, cells called that class, correct calls, precision, recall and the fraction left unclassified.
  - `threshold_summary_other<a>_state<b>.csv`: for each trained gamma and split: cells unclassified, accuracy of the classified cells, macro precision and macro recall.
  - `family_report_other<a>_state<b>.csv`: for each trained gamma, split and true class: the fraction called that state, called its family, called something wrong, or left unclassified.
  - `family_summary_other<a>_state<b>.csv`: for each trained gamma and split: the fraction unclassified (all cells and epithelial cells), the fraction of epithelial cells called at state and at family level, the accuracy of the calls, and each family call's count and precision.
  - `threshold_grid.csv`: validation macro precision, macro recall and fraction unclassified for every trained gamma x threshold pair.
  - In `figures/11/`: `posthoc_reweighting.png`, `threshold_grid_gamma<g>.png`, `threshold_confusion_gamma<g>_other<a>_state<b>.png` (test set, with an "Unclassified" column) and `family_confusion_gamma<g>_other<a>_state<b>.png` (the same, with LASP, LHS and BMYO columns).

What it does:

- **Check:** it stops if its recomputed test probabilities or validation balanced accuracy differ from what `09` saved.
- **Part A, post-hoc reweighting:** this asks whether training with class weights helps the network learn the rare states, or only shifts its decision rule. Each network's probabilities are multiplied by (1 / class frequency)^`gamma_post`, using the training set's frequencies, and the top class is the call. Negative `gamma_post` values undo a weighted network's weights. `gamma_post` is chosen per network by validation balanced accuracy, and the test set is scored once. **If the unweighted network (trained gamma 0) plus its best `gamma_post` does as well as the weighted networks, the weights only shift the decision rule.** An unweighted network's probabilities match the real class mix, so they're the ones to threshold.
- **Part B, thresholds:** a cell is called "Other" if P(Other) is at least the Other threshold, or one of the 10 states if that state's probability is at least the state threshold. If several classes pass, the most probable one wins. Otherwise the cell is "Unclassified". **With the defaults (0.6 and 0.8) no cell can pass two thresholds,** because the probabilities sum to 1. That can only happen when two thresholds sum to 1 or less.
- **Scores:** precision is correct calls ÷ cells called that class. Recall is correct calls ÷ all true cells of that class, so unclassified cells count as misses. Macro recall is then balanced accuracy with unclassified cells counted as wrong.
- **Part C, family fallback:** as Part B, but a cell left unclassified is called LASP, LHS or BMYO if that family's summed probability (e.g. LASP1 + … + LASP5) is at least the state threshold. A family call counts as right if the cell's true state is in that family. The families are `FAMILIES` in `hbca_classifier/settings.py`.
- **Threshold grid:** every pair from `THRESHOLD_GRID` (0.5 to 0.95), on validation only. **It's exploratory: choose thresholds from the grid, not from the test results,** or the test scores will look better than they are.

The baseline and KNN aren't included, because `09` didn't save their validation scores. The settings (`POSTHOC_GAMMAS`, `THRESHOLD_GRID`) are in `hbca_classifier/settings.py`. `slurm/Run_11.sh`: 4 CPUs, 32G.

### Test figures: figures.py

Each model's test figures are drawn only from its saved predictions, by `hbca_classifier/plots.py`. There are three per model:

- `<prefix>test_confusion_matrix.png`: the fraction of each true class predicted as each class. **Rows sum to 1, and the diagonal is recall.**
- `<prefix>test_confusion_matrix_precision.png`: the fraction of each predicted class that was really each class. **Columns sum to 1, and the diagonal is precision.** The rest of each column shows what the wrong calls really were.
- `<prefix>test_precision_recall.png`: precision and recall of each class side by side, with each class's number of test cells.

Each title has the model's test balanced accuracy and macro F1. To change a figure, edit `hbca_classifier/plots.py`, then redraw every model of a run:

```
sbatch slurm/Run_figures.sh [max_cells_per_class]    # e.g. 2000, or nothing for the full data
```

- **Needs:** the run's `test_predictions_<model>[_max<G>genes].npz` files (from `05`, `06`, `07`).
- **Produces:** each model's three test figures, in its step's figure folder (`figures/05/`, `06/` or `07/`, with `cap<N>/` for a capped run). It refits nothing and takes about a minute. `slurm/Run_figures.sh`: 2 CPUs, 16G.

### The model code

- `hbca_classifier/group_enet.py` fits the model with PyTorch, by accelerated proximal gradient (FISTA). Each step is a few large matrix products, so it uses every CPU core the job has, and it would also run on a GPU (`device="cuda"`). It takes optional per-cell weights (`sample_weight`). The weighted version was checked two ways: against scikit-learn's weighted ridge fit (probabilities within 1e-7), and against an unweighted fit on duplicated cells (within 1e-9).
- `dev/multinomial_group_lasso.py` is the same model for skglm. The scripts don't use it. It's kept to check the PyTorch solver. The two agreed on simulated data and on 300 real cells (probabilities within about 1e-7). **skglm is far slower:** on 20,000 cells, one alpha took up to 47 minutes with skglm, against 11 seconds with PyTorch on 16 CPUs. The model was also checked against scikit-learn: unpenalized and pure-ridge fits give the same probabilities to within 1e-7.

### The torch environment

The classifier steps need PyTorch, which is in its own conda environment, `torch` (about 7 GB):

```
conda create -y -n torch python=3.12 numpy scipy pandas scikit-learn h5py openpyxl matplotlib seaborn ipykernel pip
conda activate torch
PYTHONNOUSERSITE=1 python -m pip install torch skglm
```

**Use `python -m pip`, not `pip`.** Otherwise the cluster's system `pip` may be found first, and it installs into `~/.local` instead of the environment.

## Earlier results

Every run up to 2026-10-19 split the data by cell: training and test cells could come from the same donor. **Their outputs were moved to `<STORAGE_DIR>/old_cell_split/` and `figures/old_cell_split/`,** and the scripts no longer read them. The main ones (test balanced accuracy, 40-gene limit):

- Cap 20,000 cells per class: 25 genes, network 0.842.
- Full data, every-cell gene scaling: 25 genes, network 0.750.
- Full data, class-weighted gene scaling: 38 genes, network 0.867.
- TRANSPLANT (`09`–`11`), on the cap-20,000 genes: network 0.843.

## Other files

- `config.R`: settings for the R scripts (`figure_outdir`, `data_dir`, `OUTPUT_DIR` and `get_local_path()` from `local_paths.env`, `catObj()`, point sizes, `rng.seed`). It stays in the repo root, because the R scripts source it from there.
- `hbca_classifier/config.py`: the same paths for the Python scripts (`OUTPUT_DIR`, `get_local_path()`).
- `hbca_classifier/settings.py`: the classifier's settings (classes, split, folds, penalty grid, solver tolerance, network, KNN, UMAP and TRANSPLANT settings) and shared helpers.
- `hbca_classifier/group_enet.py`: the tier-1 model's solver (see "The model code").
- `hbca_classifier/tier2_models.py`: the tier-2 network training and KNN votes as functions, used by `09`.
- `hbca_classifier/plots.py`: saves and loads test predictions, and draws the test figures (see "Test figures").
- `local_paths.env.example`: template for `local_paths.env`.
- `slurm/Run_01.sh`: SLURM script for `01_atlas_exploration.R`. It requests 100G and 1 day on `htc`, and runs in the `renv` environment, because loading the atlas takes a lot of memory. `slurm/Run_02.sh` (for `02_marker_matrices.R`) has the same resources.
- `slurm/Run_03.sh` to `slurm/Run_11.sh`: SLURM scripts for the classifier steps (`torch` environment, except `slurm/Run_08.sh`: base). Each puts the repo root on `PYTHONPATH`.
- `logs/`: holds a `.gitkeep`, because SLURM fails a job whose log folder is missing.
