# hbca_classifier/settings.py
#
# Settings and helpers shared by the three level2 classifier steps:
#   03_prep.py -> 04_cv.py (SLURM array) -> 05_fit.py
# Change the settings here, not in the scripts, so all three steps agree.
#
# Every step takes the same optional first argument: [max_cells_per_class]
#   max_cells_per_class:  default none (every cell). Caps the cells per class before the split, for a
#                         quicker run. Each cap gets its own output folder, so capped runs never
#                         overwrite the full one. Pass 'none' for no cap when a later argument follows.
# 05 to 08 also take a second, max_genes (see 05_fit.py); 09 and 10 take none; 11 takes two thresholds.

import os
import sys

import numpy as np
import pandas as pd

# Shared Python config (hbca_classifier/config.py): OUTPUT_DIR (the storage folder, from local_paths.env)
from hbca_classifier.config import OUTPUT_DIR

# ---------------------- Settings
# The classes to predict. Every other level2 label becomes "Other".
KEEP_STATES = ['LASP1', 'LASP2', 'LASP3', 'LASP4', 'LASP5',
               'LHS1', 'LHS2', 'LHS3',
               'BMYO1', 'BMYO2']
CLASSES = np.array(KEEP_STATES + ['Other'])      # fixed class order for coefficients and plots
# The states' families (11 falls back to these when no single state is confident enough)
FAMILIES = {'LASP': KEEP_STATES[:5], 'LHS': KEEP_STATES[5:8], 'BMYO': KEEP_STATES[8:]}

TEST_SIZE = 0.3          # 70/30 train/test split
N_FOLDS = 5              # cross-validation folds (on the training set)
RANDOM_STATE = 0

# Penalty grid. l1_ratio = 1 is pure group lasso, 0 would be pure ridge.
# For each l1_ratio, N_ALPHAS values evenly spaced (on a linear scale) from just below alpha_max down
# to ALPHA_MIN. alpha_max is the smallest alpha that drops every gene, so it is left out (that model
# is empty). Linear spacing puts most values near the top, where models keep few genes, and fewer
# toward ALPHA_MIN, where they keep many. alpha_max depends on l1_ratio and the data (set in 03).
# 04 runs one SLURM array task per (fold, l1_ratio): N_FOLDS * len(L1_RATIOS) tasks.
# **If you change N_FOLDS or L1_RATIOS, change the --array line in slurm/Run_04.sh to match.**
L1_RATIOS = [0.25, 0.5, 0.75, 1.0]
N_ALPHAS = 20
ALPHA_MIN = 0.01

# Gene scaling for tier 1 (03-05): each gene is divided by its SD on the training cells.
#   'cells':    every cell counts equally (the original runs)
#   'balanced': cells are weighted by inverse class frequency, as in the loss, so every class counts
#               equally. In the full data, 57% of cells are Other, so with 'cells' a gene expressed only
#               in a small population gets a tiny SD, is scaled up a lot, and becomes cheap to keep.
# Chosen per run with the HBCA_SCALING environment variable, e.g.
#   HBCA_SCALING=balanced sbatch slurm/Run_03.sh 2000
# A balanced run gets its own classifier and figure folders (_balanced), so it never overwrites a
# 'cells' run. **Set it the same way for every step of a run (03-08), and leave it unset for 09-11.**
GENE_SCALING = os.environ.get('HBCA_SCALING', 'cells')
if GENE_SCALING not in ('cells', 'balanced'):
    raise ValueError(f"HBCA_SCALING must be 'cells' or 'balanced', not {GENE_SCALING!r}")

# Solver settings (FISTA in hbca_classifier/group_enet.py)
TOL = 1e-6               # stop when the largest optimality violation is below this
MAX_ITER = 20000         # per alpha

# Neural network on the selected genes (06). Trained on the training set minus a validation part,
# which decides when to stop (early stopping). The loss is weighted by inverse class frequency.
NN_VAL_SIZE = 0.1        # fraction of the training set held back for validation
NN_HIDDEN = [64, 32]     # units in each hidden layer
# Other network shapes 06 can train instead, chosen by name (its 3rd argument). Their outputs get a
# _<name> suffix, so they never overwrite the default network's.
NN_ARCHITECTURES = {
    'deep': [64, 64, 32, 32, 16],
}
NN_DROPOUT = 0.1
NN_LEARNING_RATE = 1e-3  # Adam
NN_BATCH_SIZE = 512
NN_MAX_EPOCHS = 500
NN_PATIENCE = 20         # stop after this many epochs without a better validation loss

# Weighted KNN on the selected genes (07). Same training / validation / test cells as 06.
# Each neighbour's vote is weighted by 1 / distance times (1 / frequency of its class)^gamma.
# Each gene is centred and divided by SD^beta (beta = 1: unit variance; 0.5: Pareto scaling; 0: no
# scaling), which sets how much each gene counts in the distance.
# k, gamma and beta are tuned together on the validation set, by balanced accuracy.
KNN_K = [5, 10, 25, 50, 100, 200, 400]
KNN_GAMMAS = [0, 0.25, 0.5, 0.75, 1]   # 0 = plain vote, 1 = full inverse-frequency class weights
KNN_BETAS = [0, 0.5, 1]                # gene scaling: divide by SD^beta
KNN_QUERY_BATCH = 256                  # query cells per batch in the neighbour search (memory)

# UMAP of the atlas cells on the selected genes (08)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.3
UMAP_OTHER_SAMPLE = 100000   # "Other" cells to include (a random sample); every epithelial cell is included


# TRANSPLANT (09, 10): take the genes from a capped run's tier-1 model and retrain the tier-2 models
# on the full data, with tempered class weights (1/class frequency)^gamma, gamma tuned on validation.
# **If you change TRANSPLANT_GAMMAS, change the --array line in slurm/Run_09.sh to match** (one task per gamma, plus KNN).
TRANSPLANT_SOURCE_CAP = 20000
TRANSPLANT_SOURCE_MAX_GENES = 40
TRANSPLANT_GAMMAS = [0, 0.25, 0.5, 0.75, 1]

# Decision rules for the TRANSPLANT networks (11)
# Post-hoc reweighting: probabilities x (1/class frequency)^gamma_post, then the top class.
# Negative values undo a weighted network's class weights.
POSTHOC_GAMMAS = [round(g, 3) for g in np.arange(-1, 1.5001, 0.125)]
# Threshold pairs (Other threshold x state threshold) scored on validation
THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.5, 0.9501, 0.05)]

# ---------------------- Helpers
def get_args():
    """Read [max_cells_per_class] (the first command-line argument) from the command line."""
    return int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] not in ('', 'none') else None


def get_classifier_dir(cap):
    """Folder for a run's classifier files: <STORAGE_DIR>/reed_level2_classifier[_cap<N>][_balanced]/"""
    name = 'reed_level2_classifier' + ('' if cap is None else f'_cap{cap}') \
        + ('_balanced' if GENE_SCALING == 'balanced' else '')
    d = os.path.join(OUTPUT_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def get_figure_dir(cap, step):
    """figures/<step>/, or figures/<step>/cap<N>/ for a capped run; with balanced gene scaling,
    figures/<step>/balanced/ or figures/<step>/cap<N>_balanced/."""
    sub = ('' if cap is None else f'cap{cap}') + ('_balanced' if GENE_SCALING == 'balanced' else '')
    d = os.path.join('figures', step, *([sub.lstrip('_')] if sub else []))
    os.makedirs(d, exist_ok=True)
    return d


def load_prepared(clf_dir):
    """Read what 03 saved: the cells x genes matrix (unscaled, CSR) and a dict with the labels,
    the split, the folds and the alpha grids."""
    import scipy.sparse as sp
    X = sp.load_npz(os.path.join(clf_dir, 'X.npz')).tocsr()
    with np.load(os.path.join(clf_dir, 'prep.npz')) as f:
        prep = {k: f[k] for k in f.files}
    return X, prep


def one_hot(y):
    """One-hot encode labels in the fixed class order."""
    return (y[:, None] == CLASSES[None, :]).astype(np.float64)


def get_class_weights(y):
    """Inverse-frequency weight per class (in the order of CLASSES), from labels y, scaled so the
    average cell has weight 1. With these weights every class counts equally in the loss."""
    counts = pd.Series(y).value_counts().reindex(CLASSES).fillna(0).to_numpy()
    return len(y) / (len(CLASSES) * np.maximum(counts, 1))


def get_sample_weights(y, class_weights):
    """Each cell's weight: its class's weight."""
    return class_weights[pd.Index(CLASSES).get_indexer(y)]


def get_scaling_weights(y):
    """Cell weights for the gene scaler: None for 'cells' scaling (every cell equal), or each cell's
    inverse-class-frequency weight for 'balanced' scaling."""
    return None if GENE_SCALING == 'cells' else get_sample_weights(y, get_class_weights(y))


def mean_log_loss(y, P, class_weights=None):
    """Mean cross-entropy of true labels y under probabilities P (columns in the order of CLASSES).
    With class_weights, a weighted mean, so every class counts equally (like balanced accuracy).
    sklearn's log_loss assumes alphabetical column order, which CLASSES is not."""
    col = pd.Index(CLASSES).get_indexer(y)
    losses = -np.log(np.clip(P[np.arange(len(y)), col], 1e-15, None))
    weights = None if class_weights is None else class_weights[col]
    return np.average(losses, weights=weights)


def set_threads():
    """Let PyTorch use every CPU core the SLURM job was given."""
    import torch
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    torch.set_num_threads(threads)
    return threads
