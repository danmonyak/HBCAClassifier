# 04_cv.py
#
# Step 2 of 3 of the level2 classifier: one cross-validation fit, run as a SLURM array task
# (slurm/Run_04.sh). Each task takes one (fold, l1_ratio) pair: it fits the model on the other folds
# for every alpha in the grid (largest first, each fit warm-started from the previous one) and
# scores the held-out fold. 5 folds x 4 l1_ratios = 20 tasks.
#
# The model is multinomial logistic regression with a group elastic net penalty (each gene's
# coefficients across all classes form one group), fitted with PyTorch on all the job's CPU
# cores (hbca_classifier/group_enet.py). Genes are scaled to unit variance on the fold's
# training part only (not centred, which keeps the matrix sparse); with HBCA_SCALING=balanced, the
# variance is class-weighted (GENE_SCALING in hbca_classifier/settings.py). The intercept isn't penalized.
# Cells are weighted by inverse class frequency, so rare classes (e.g. LASP5) count as much as
# common ones, in the fit and in the held-out log-loss.
#
# Needs:    X.npz and prep.npz from 03, in <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/
#           SLURM_ARRAY_TASK_ID (set by SLURM; set it by hand to run one task outside an array)
# Produces: cv/fold<k>_l1ratio<r>.csv in the same folder: one row per alpha, with the held-out
#           log-loss, accuracy, balanced accuracy, macro F1, genes kept, and the solver's convergence
#
# Usage: sbatch slurm/Run_04.sh [max_cells_per_class]
#        (python scripts/04_cv.py [max_cells_per_class])

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from hbca_classifier.settings import (CLASSES, N_FOLDS, L1_RATIOS, TOL, MAX_ITER,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared,
                                      one_hot, mean_log_loss, set_threads,
                                      get_class_weights, get_sample_weights, get_scaling_weights, GENE_SCALING)
from hbca_classifier.group_enet import MultinomialGroupEnet, predict_proba

# torch warns that sparse CSR support is in beta; it doesn't affect the results
warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta.*')

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

max_cells_per_class = get_args()
clf_dir = get_classifier_dir(max_cells_per_class)
get_figure_dir(max_cells_per_class, '04')   # no figures; made like every step
cv_dir = os.path.join(clf_dir, 'cv')
os.makedirs(cv_dir, exist_ok=True)

# ---------------------- Which (fold, l1_ratio) this task fits
task = int(os.environ['SLURM_ARRAY_TASK_ID'])
n_tasks = N_FOLDS * len(L1_RATIOS)
if not 0 <= task < n_tasks:
    raise ValueError(f"array task {task} is out of range: there are {n_tasks} tasks (0-{n_tasks - 1}). "
                     "Check that the --array line in slurm/Run_04.sh matches hbca_classifier/settings.py.")
k, l1_index = divmod(task, len(L1_RATIOS))
l1_ratio = L1_RATIOS[l1_index]
threads = set_threads()
print(f"Max cells per class: {max_cells_per_class}, gene scaling: {GENE_SCALING}")
print(f"Task {task}: fold {k}, l1_ratio {l1_ratio}, {threads} threads\n")

# ---------------------- Data for this fold
X, prep = load_prepared(clf_dir)
assert np.allclose(prep['l1_ratios'], L1_RATIOS), "L1_RATIOS changed since 03 ran: rerun 03"
alphas = prep['alpha_grids'][l1_index]

idx_train, fold = prep['idx_train'], prep['fold']
tr, va = idx_train[fold != k], idx_train[fold == k]
y_tr, y_va = prep['y'][tr], prep['y'][va]

scaler = StandardScaler(with_mean=False).fit(X[tr], sample_weight=get_scaling_weights(y_tr))
X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])
del X
log(f"fold {k}: fitting on {len(tr):,} cells, scoring on {len(va):,}")

# Weight cells by inverse class frequency (in this fold's training part), so every class counts
# equally in the fit and in the held-out log-loss
class_weights = get_class_weights(y_tr)
model = MultinomialGroupEnet(X_tr, one_hot(y_tr), sample_weight=get_sample_weights(y_tr, class_weights))
log("model set up")

# ---------------------- Fit the alpha path and score the held-out fold
rows = []
W, b = None, None
for a_index, alpha in enumerate(alphas):
    t = time.time()
    W, b, info = model.fit(alpha, l1_ratio, W, b, tol=TOL, max_iter=MAX_ITER)
    P = predict_proba(X_va, W, b)
    pred = CLASSES[P.argmax(axis=1)]
    rows.append(dict(
        fold=k, l1_ratio=l1_ratio, alpha=alpha,
        log_loss=mean_log_loss(y_va, P, class_weights),
        accuracy=accuracy_score(y_va, pred),
        balanced_accuracy=balanced_accuracy_score(y_va, pred),
        # macro F1: the mean of the per-class F1 scores, so every class counts equally
        f1_macro=f1_score(y_va, pred, labels=CLASSES, average='macro', zero_division=0),
        n_genes=int((np.linalg.norm(W, axis=1) > 0).sum()),
        n_iter=info['n_iter'], kkt=info['kkt'], converged=info['converged'],
        seconds=time.time() - t,
    ))
    r = rows[-1]
    log(f"alpha {a_index + 1}/{len(alphas)} = {alpha:.3g}: {r['seconds']:.1f} s, {r['n_iter']} iterations"
        f"{'' if r['converged'] else ' (NOT converged)'}, {r['n_genes']} genes, "
        f"log-loss {r['log_loss']:.4f}, balanced accuracy {r['balanced_accuracy']:.3f}, macro F1 {r['f1_macro']:.3f}")

out_file = os.path.join(cv_dir, f'fold{k}_l1ratio{l1_ratio}.csv')
pd.DataFrame(rows).to_csv(out_file, index=False)
log(f"saved {out_file}")
