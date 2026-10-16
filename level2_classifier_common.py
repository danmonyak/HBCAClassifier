# level2_classifier_common.py
#
# Settings and helpers shared by the three level2 classifier steps:
#   03_Level2_Classifier_Prep.py -> 04_Level2_Classifier_CV.py (SLURM array) -> 05_Level2_Classifier_Fit.py
# Change the settings here, not in the scripts, so all three steps agree.
#
# Every step takes the same optional arguments: [cellranger_run_name] [max_cells_per_class]
#   cellranger_run_name:  default 90. Inputs are read from <STORAGE_DIR>/<run>/.
#   max_cells_per_class:  default none (every cell). Caps the cells per class before the split, for a
#                         quicker run. Each cap gets its own output folder, so capped runs never
#                         overwrite the full one. Pass 'none' for no cap when a later argument follows.
# 05 also takes a third, max_genes (see 05_Level2_Classifier_Fit.py).

import os
import sys

import numpy as np
import pandas as pd

# Shared Python config (RNAseq_SingleCell/config.py): OUTPUT_DIR (the storage folder, from local_paths.env)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# ---------------------- Settings
# The classes to predict. Every other level2 label becomes "Other".
KEEP_STATES = ['LASP1', 'LASP2', 'LASP3', 'LASP4', 'LASP5',
               'LHS1', 'LHS2', 'LHS3',
               'BMYO1', 'BMYO2']
CLASSES = np.array(KEEP_STATES + ['Other'])      # fixed class order for coefficients and plots

TEST_SIZE = 0.3          # 70/30 train/test split
N_FOLDS = 5              # cross-validation folds (on the training set)
RANDOM_STATE = 0

# Penalty grid. l1_ratio = 1 is pure group lasso, 0 would be pure ridge.
# For each l1_ratio, N_ALPHAS values evenly spaced (on a linear scale) from just below alpha_max down
# to ALPHA_MIN. alpha_max is the smallest alpha that drops every gene, so it is left out (that model
# is empty). Linear spacing puts most values near the top, where models keep few genes, and fewer
# toward ALPHA_MIN, where they keep many. alpha_max depends on l1_ratio and the data (set in 03).
# 04 runs one SLURM array task per (fold, l1_ratio): N_FOLDS * len(L1_RATIOS) tasks.
# **If you change N_FOLDS or L1_RATIOS, change the --array line in Run_04.sh to match.**
L1_RATIOS = [0.25, 0.5, 0.75, 1.0]
N_ALPHAS = 10
ALPHA_MIN = 0.01

# Solver settings (FISTA in multinomial_group_enet_torch.py)
TOL = 1e-6               # stop when the largest optimality violation is below this
MAX_ITER = 20000         # per alpha

# Neural network on the selected genes (06). Trained on the training set minus a validation part,
# which decides when to stop (early stopping). The loss is weighted by inverse class frequency.
NN_VAL_SIZE = 0.1        # fraction of the training set held back for validation
NN_HIDDEN = [64, 32]     # units in each hidden layer
NN_DROPOUT = 0.1
NN_LEARNING_RATE = 1e-3  # Adam
NN_BATCH_SIZE = 512
NN_MAX_EPOCHS = 500
NN_PATIENCE = 20         # stop after this many epochs without a better validation loss


# ---------------------- Helpers
def get_args():
    """Read [cellranger_run_name] [max_cells_per_class] from the command line."""
    run = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else '90'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ('', 'none') else None
    return run, cap


def get_classifier_dir(run, cap):
    """Folder for this run's classifier files: <STORAGE_DIR>/<run>/reed_level2_classifier[_cap<N>]/"""
    name = 'reed_level2_classifier' + ('' if cap is None else f'_cap{cap}')
    d = os.path.join(OUTPUT_DIR, run, name)
    os.makedirs(d, exist_ok=True)
    return d


def get_figure_dir(run, cap, step):
    """figures/<run>/<step>/, or figures/<run>/<step>/cap<N>/ for a capped run."""
    d = os.path.join('figures', run, step, *([] if cap is None else [f'cap{cap}']))
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


def mean_log_loss(y, P):
    """Mean cross-entropy of true labels y under probabilities P (columns in the order of CLASSES).
    sklearn's log_loss assumes alphabetical column order, which CLASSES is not."""
    col = pd.Index(CLASSES).get_indexer(y)
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), col], 1e-15, None)))


def set_threads():
    """Let PyTorch use every CPU core the SLURM job was given."""
    import torch
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    torch.set_num_threads(threads)
    return threads
