# 07_knn.py
#
# A second tier-2 model: weighted k-nearest neighbours on the genes the tier-1 model (05) selected.
# It uses the same cells as 06: the training set from 03 minus the same validation donors, which
# is the reference, and the validation donors' cells, on which k and the class weighting are
# tuned. The test set is used once, at the end.
#
# Each neighbour votes for its class with weight 1 / distance times (1 / frequency of its class
# in the reference)^gamma. gamma = 0 is a plain distance-weighted vote; gamma = 1 fully corrects
# for class imbalance. Each gene is centred and divided by SD^beta, using the reference cells:
# beta = 1 is unit variance, 0.5 is Pareto scaling, 0 is no scaling (each gene keeps its own
# spread, so high-variance genes count more in the distance). k, gamma and beta are tuned together
# by validation balanced accuracy. Settings (KNN_K, KNN_GAMMAS, KNN_BETAS) are in
# hbca_classifier/settings.py.
#
# For each beta, the neighbours are found once per cell, at the largest k (brute force, in batches,
# with PyTorch on all the job's CPUs), and every (k, gamma) is scored from the same neighbour lists.
#
# Needs:    from 03 and 05 (run with the same arguments), in
#           <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/: X.npz, prep.npz, model[_max<G>genes].npz
#           Optional: nn_comparison[_max<G>genes].csv from 06, to compare with the network and baseline.
# Produces: in that folder (all with the _max<G>genes suffix when max_genes is given):
#           knn_tuning.csv       validation balanced accuracy and macro F1 for every (k, gamma, beta)
#           knn_test_report.csv  per-class precision, recall and F1 on the test set
#           knn_comparison.csv   test accuracy, balanced accuracy and macro F1 for KNN, plus the
#                                network and baseline rows from 06 if they exist
#           in figures/07/ (or .../07/cap<N>/): knn_tuning.png (validation balanced accuracy
#           over k x gamma, one panel per beta) and knn_test_confusion_matrix.png
#
# Usage: python scripts/07_knn.py [max_cells_per_class] [max_genes]
#        e.g. after `sbatch slurm/Run_05.sh 20000 40`: sbatch slurm/Run_07.sh 20000 40

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from hbca_classifier.settings import (score_text, donor_split, describe_split, CLASSES, RANDOM_STATE, NN_VAL_SIZE, KNN_K, KNN_GAMMAS, KNN_BETAS, KNN_QUERY_BATCH,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared, set_threads)

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

max_cells_per_class = get_args()
max_genes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ('', 'none') else None
suffix = '' if max_genes is None else f'_max{max_genes}genes'     # matches 05's output names
clf_dir = get_classifier_dir(max_cells_per_class)
figure_outdir_cur = get_figure_dir(max_cells_per_class, '07')
threads = set_threads()
print(f"Max cells per class: {max_cells_per_class}, "
      f"max genes: {max_genes}, {threads} threads\n")

# ---------------------- The genes selected by the tier-1 model (05)
model_file = os.path.join(clf_dir, f'model{suffix}.npz')
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found: run 05 with the same arguments first")
with np.load(model_file) as f:
    selected = np.linalg.norm(f['W'], axis=1) > 0
    genes = f['genes'][selected]
print(f"{len(genes)} genes selected by 05: {', '.join(genes)}\n")

# ---------------------- Data: the same split as 06
X, prep = load_prepared(clf_dir)
X = X[:, np.flatnonzero(selected)].toarray()
y = prep['y']
idx_test = prep['idx_test']
# Validation: the cells of about NN_VAL_SIZE of the training donors (the same ones as in 06)
idx_train = prep['idx_train']
is_val, _ = donor_split(prep['donors'][idx_train], y[idx_train], NN_VAL_SIZE)
idx_ref, idx_val = idx_train[~is_val], idx_train[is_val]
describe_split(y[idx_train], is_val, prep['donors'][idx_train], 'Validation')
X_ref_raw, X_val_raw, X_test_raw = X[idx_ref], X[idx_val], X[idx_test]
y_ref, y_val, y_test = y[idx_ref], y[idx_val], y[idx_test]
del X
gene_mean, gene_sd = X_ref_raw.mean(axis=0), X_ref_raw.std(axis=0)
gene_sd[gene_sd == 0] = 1.0     # a gene with no spread in the reference: leave it unscaled


def scale(X_raw, beta):
    """Centre each gene and divide it by SD^beta (from the reference cells), as a float32 tensor."""
    return torch.as_tensor((X_raw - gene_mean) / gene_sd ** beta, dtype=torch.float32)


log(f"reference: {len(idx_ref):,} cells, validation: {len(idx_val):,}, test: {len(idx_test):,}")

k_max = max(KNN_K)
if k_max > len(idx_ref):
    raise ValueError(f"largest k ({k_max}) is more than the reference cells ({len(idx_ref)})")

ref_class = torch.as_tensor(pd.Index(CLASSES).get_indexer(y_ref), dtype=torch.long)
ref_freq = np.bincount(ref_class.numpy(), minlength=len(CLASSES)) / len(y_ref)


def class_vote_sums(Xq, X_ref):
    """For every k in KNN_K: an (n_query x n_classes) array of each class's distance-weighted votes
    among the k nearest reference cells (class weights are applied later, per gamma)."""
    ref_sq = (X_ref ** 2).sum(dim=1)
    sums = {k: np.empty((len(Xq), len(CLASSES)), dtype=np.float64) for k in KNN_K}
    for start in range(0, len(Xq), KNN_QUERY_BATCH):
        q = Xq[start:start + KNN_QUERY_BATCH]
        # squared Euclidean distances to every reference cell: |q|^2 + |r|^2 - 2 q.r
        d2 = (q ** 2).sum(dim=1, keepdim=True) + ref_sq - 2 * q @ X_ref.T
        d2, nn_idx = torch.topk(d2, k_max, dim=1, largest=False, sorted=True)
        # 1 / distance. Exact matches (distance 0) get a very large weight, so they dominate,
        # as in scikit-learn's weights="distance".
        w = 1 / torch.sqrt(d2.clamp(min=0)).clamp(min=1e-6)
        votes = torch.zeros(len(q), k_max, len(CLASSES))
        votes.scatter_(2, ref_class[nn_idx].unsqueeze(2), w.unsqueeze(2))
        cum = votes.cumsum(dim=1)                          # votes among the first j neighbours
        for k in KNN_K:
            sums[k][start:start + len(q)] = cum[:, k - 1, :].double().numpy()
    return sums


def predict(sums_k, gamma):
    """Class scores (rows sum to 1) and predicted labels for one k and gamma."""
    scores = sums_k * (1 / ref_freq) ** gamma
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores, CLASSES[scores.argmax(axis=1)]


# ---------------------- Tune k, gamma and beta on the validation set
rows = []
for beta in KNN_BETAS:
    val_sums = class_vote_sums(scale(X_val_raw, beta), scale(X_ref_raw, beta))
    log(f"beta {beta}: validation neighbours found")
    for k in KNN_K:
        for gamma in KNN_GAMMAS:
            _, pred = predict(val_sums[k], gamma)
            rows.append(dict(beta=beta, k=k, gamma=gamma, balanced_accuracy=balanced_accuracy_score(y_val, pred),
                             f1_macro=f1_score(y_val, pred, labels=CLASSES, average='macro', zero_division=0)))
tuning = pd.DataFrame(rows)
tuning.to_csv(os.path.join(clf_dir, f'knn_tuning{suffix}.csv'), index=False)
best = tuning.loc[tuning['balanced_accuracy'].idxmax()]
best_k, best_gamma, best_beta = int(best['k']), float(best['gamma']), float(best['beta'])
print(f"\nBest on validation: k = {best_k}, gamma = {best_gamma}, beta = {best_beta} "
      f"(balanced accuracy {best['balanced_accuracy']:.4f}, macro F1 {best['f1_macro']:.4f})")
print("Best validation balanced accuracy for each beta:")
print(tuning.groupby('beta')['balanced_accuracy'].max().round(4).to_string())
if best_k in (min(KNN_K), max(KNN_K)):
    print(f"**Note:** the best k is at the edge of KNN_K ({best_k}); a wider grid may do better.")

# Heatmaps of validation balanced accuracy over k x gamma, one panel per beta, on a shared colour scale
fig, axes = plt.subplots(1, len(KNN_BETAS), figsize=(7 * len(KNN_BETAS), 4), squeeze=False)
vmin, vmax = tuning['balanced_accuracy'].min(), tuning['balanced_accuracy'].max()
for ax, (beta, d) in zip(axes[0], tuning.groupby('beta')):
    sns.heatmap(d.pivot(index='gamma', columns='k', values='balanced_accuracy'), annot=True, fmt='.3f',
                cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': 'Validation balanced accuracy'}, ax=ax)
    ax.set(title=f'beta = {beta}' + {0: ' (no scaling)', 0.5: ' (Pareto)', 1: ' (unit variance)'}.get(beta, ''))
fig.suptitle(f'Weighted KNN tuning, {len(genes)} genes (best: k = {best_k}, gamma = {best_gamma}, beta = {best_beta})')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'knn_tuning{suffix}.png'), dpi=200)
plt.close(fig)

# ---------------------- Evaluate the chosen k and gamma on the test set
test_sums = class_vote_sums(scale(X_test_raw, best_beta), scale(X_ref_raw, best_beta))
log("test neighbours found")
_, pred_test = predict(test_sums[best_k], best_gamma)
print("\nWeighted KNN on the test set:")
print(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0))
pd.DataFrame(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0, output_dict=True)) \
    .T.to_csv(os.path.join(clf_dir, f'knn_test_report{suffix}.csv'))

cm = confusion_matrix(y_test, pred_test, labels=CLASSES, normalize='true')
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
            vmin=0, vmax=1, cbar_kws={'label': 'Fraction of true class'}, ax=ax)
ax.set(xlabel='Predicted', ylabel='True',
       title=f'Weighted KNN, {len(genes)} genes (k = {best_k}, gamma = {best_gamma}, beta = {best_beta}): test set\n'
             f'{score_text(y_test, pred_test)}')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'knn_test_confusion_matrix{suffix}.png'), dpi=200)
plt.close(fig)

# ---------------------- Compare with 06 (network and baseline), if it has run
# KNN has no log-loss: its probabilities are vote shares, which move in steps of about 1/k.
knn_row = dict(model=f'Weighted KNN (k = {best_k}, gamma = {best_gamma}, beta = {best_beta})', log_loss=np.nan,
               accuracy=accuracy_score(y_test, pred_test),
               balanced_accuracy=balanced_accuracy_score(y_test, pred_test),
               f1_macro=f1_score(y_test, pred_test, labels=CLASSES, average='macro', zero_division=0))
comparison = pd.DataFrame([knn_row])
nn_file = os.path.join(clf_dir, f'nn_comparison{suffix}.csv')
if os.path.exists(nn_file):
    comparison = pd.concat([pd.read_csv(nn_file), comparison], ignore_index=True)
else:
    print(f"({nn_file} not found: run 06 to compare with the network and baseline)")
comparison.to_csv(os.path.join(clf_dir, f'knn_comparison{suffix}.csv'), index=False)
print("\nTest set:")
print(comparison.round(4).to_string(index=False))
log("done")
