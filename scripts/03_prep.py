# 03_prep.py
#
# Step 1 of 3 of the level2 classifier: prepares the data for cross-validation (04) and the
# final fit (05). Reads the cell state marker matrix and the labels from 02, relabels every
# non-epithelial state as "Other", makes the train/test split and the 5 CV folds, and sets the
# alpha grid for each l1_ratio. Settings are in hbca_classifier/settings.py.
#
# Both splits are by donor: every donor's cells are all in training or all in test, and all in one
# CV fold. The test donors are about 30% of the donors, chosen so each class has about 30% of its
# cells in test (donor_split). The folds come from StratifiedGroupKFold. Each cell's donor is read
# from the atlas (obs/donor_id in hbca-global.h5ad).
#
# Needs:    from 02_marker_matrices.R, in <STORAGE_DIR>/:
#           reed_state_marker_expr.mtx.gz, .genes.txt.gz, .cells.txt.gz, reed_cell_labels.csv.gz
#           hbca-global.h5ad (in HBCA_DIR), for each cell's donor
# Produces: in <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/:
#           X.npz     cells x genes log-normalized expression (unscaled, sparse), after any cap
#           prep.npz  cells, genes, labels (y), donors, train/test indices, CV fold of each
#                     training cell, and the alpha grid for each l1_ratio
#
# Usage: python scripts/03_prep.py [max_cells_per_class]

import os
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from hbca_classifier.config import get_local_path
from hbca_classifier.settings import (OUTPUT_DIR, KEEP_STATES, CLASSES, TEST_SIZE, N_FOLDS, RANDOM_STATE,
                                      donor_split, describe_split,
                                      L1_RATIOS, N_ALPHAS, ALPHA_MIN,
                                      get_args, get_classifier_dir, get_figure_dir, one_hot,
                                      get_class_weights, get_sample_weights, get_scaling_weights)

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

max_cells_per_class = get_args()
data_dir = OUTPUT_DIR       # where 02 saved its files
clf_dir = get_classifier_dir(max_cells_per_class)
get_figure_dir(max_cells_per_class, '03')   # no figures yet; made like every step
print(f"Max cells per class: {max_cells_per_class}")
print(f"Output folder: {clf_dir}\n")

# ---------------------- Load the cell state marker matrix and the labels
prefix = os.path.join(data_dir, 'reed_state_marker_expr')

# 02 saved the matrix as genes x cells; the model needs cells x genes
X_all = sp.csr_matrix(mmread(prefix + '.mtx.gz', spmatrix=False).T)
genes = pd.read_csv(prefix + '.genes.txt.gz', header=None)[0].to_numpy(dtype=str)
cells = pd.read_csv(prefix + '.cells.txt.gz', header=None)[0].to_numpy(dtype=str)
labels = pd.read_csv(os.path.join(data_dir, 'reed_cell_labels.csv.gz'))

# The label file and the matrix columns must list the cells in the same order
assert X_all.shape == (len(cells), len(genes))
assert (labels['cell'].astype(str).to_numpy() == cells).all()
log(f"loaded {X_all.shape[0]:,} cells x {X_all.shape[1]:,} marker genes, {X_all.nnz:,} nonzeros")


# Each cell's donor, from the atlas's obs table (read with h5py, without loading the expression)
def read_obs_column(f, name):
    col = f['obs'][name]
    if isinstance(col, h5py.Group):     # categorical: codes into categories
        return col['categories'].asstr()[...][col['codes'][...]]
    return col.asstr()[...] if col.dtype.kind in 'OS' else col[...]


with h5py.File(os.path.join(get_local_path('HBCA_DIR'), 'hbca-global.h5ad'), 'r') as f:
    obs_names = read_obs_column(f, f['obs'].attrs['_index'])
    obs_donor = read_obs_column(f, 'donor_id')
donor_of = pd.Series(obs_donor, index=obs_names)
assert donor_of.index.is_unique and pd.Index(cells).isin(donor_of.index).all(), "cells missing from the atlas obs"
donors_all = donor_of.loc[cells].to_numpy(dtype=str)
log(f"donors: {len(np.unique(donors_all))}")

# Target: the level2 cell state for the epithelial states we care about, "Other" for everything else
y_all = np.where(labels['level2'].isin(KEEP_STATES), labels['level2'], 'Other').astype(str)
print(pd.Series(y_all).value_counts().reindex(CLASSES).to_frame('cells'))

# ---------------------- Optional cap on the cells per class
rng = np.random.default_rng(RANDOM_STATE)
if max_cells_per_class is not None:
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y_all == c), size=min(max_cells_per_class, (y_all == c).sum()), replace=False)
        for c in CLASSES
    ])
    keep.sort()
    X_all, y_all, cells, donors_all = X_all[keep], y_all[keep], cells[keep], donors_all[keep]
    log(f"capped to {len(keep):,} cells")

# ---------------------- Train/test split and CV folds, by donor
is_test, gap = donor_split(donors_all, y_all, TEST_SIZE)
idx_train, idx_test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
describe_split(y_all, is_test, donors_all, 'Test set')
print(f"  (largest gap from {TEST_SIZE} in a class's share: {gap:.3f})")
print("  test donors: " + ", ".join(sorted(np.unique(donors_all[is_test]))))
y_train, donors_train = y_all[idx_train], donors_all[idx_train]

# fold[i] is the CV fold in which training cell idx_train[i] is held out. Stratified by class,
# grouped by donor: each donor's cells are all in one fold.
fold = np.empty(len(idx_train), dtype=np.int64)
folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for k, (_, va) in enumerate(folds.split(idx_train, y_train, groups=donors_train)):
    fold[va] = k
for k in range(N_FOLDS):
    describe_split(y_train, fold == k, donors_train, f'CV fold {k}')
log(f"train: {len(idx_train):,} cells, test: {len(idx_test):,} cells, {N_FOLDS} folds")

# ---------------------- Alpha grids
# One grid per l1_ratio, from the whole (scaled) training set, shared by all folds so their
# results can be averaged. alpha_max is the smallest alpha that drops every gene: there the
# model is the intercept alone, which predicts the (class-weighted) class frequencies. Cells are
# weighted by inverse class frequency, as in the fits (04, 05).
X_train = StandardScaler(with_mean=False).fit_transform(X_all[idx_train], sample_weight=get_scaling_weights(y_train))
Y_train = one_hot(y_train)
w = get_sample_weights(y_train, get_class_weights(y_train))[:, None]
P0 = np.tile((w * Y_train).sum(axis=0) / w.sum(), (X_train.shape[0], 1))
grad_norms = np.linalg.norm(np.asarray(X_train.T @ (w * (P0 - Y_train))) / X_train.shape[0], axis=1)
# Each grid: N_ALPHAS values evenly spaced from just below alpha_max down to ALPHA_MIN (alpha_max
# itself, the empty model, is left out)
alpha_maxes = grad_norms.max() / np.array(L1_RATIOS)
if (alpha_maxes <= ALPHA_MIN).any():
    raise ValueError(f"alpha_max ({alpha_maxes.min():.3g}) is below ALPHA_MIN ({ALPHA_MIN}): lower ALPHA_MIN")
alpha_grids = np.array([a_max - (a_max - ALPHA_MIN) * np.arange(1, N_ALPHAS + 1) / N_ALPHAS
                        for a_max in alpha_maxes])
for l1r, a_max, grid in zip(L1_RATIOS, alpha_maxes, alpha_grids):
    print(f"l1_ratio {l1r}: alpha_max {a_max:.3g}, alphas " + " ".join(f"{a:.3g}" for a in grid))

# ---------------------- Save
sp.save_npz(os.path.join(clf_dir, 'X.npz'), X_all, compressed=False)    # uncompressed loads much faster
np.savez(os.path.join(clf_dir, 'prep.npz'),
         cells=cells, genes=genes, y=y_all, donors=donors_all, idx_train=idx_train, idx_test=idx_test, fold=fold,
         l1_ratios=np.array(L1_RATIOS), alpha_grids=alpha_grids)
log("saved X.npz and prep.npz")
