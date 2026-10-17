# 09_Transplant_Tier2.py
#
# TRANSPLANT, step 1 of 2: retrain the tier-2 models on the full data, on the genes that a capped
# run's tier-1 model selected (TRANSPLANT_SOURCE_CAP / TRANSPLANT_SOURCE_MAX_GENES; by default the
# 25 genes from the 20,000-per-class run with a 40-gene limit). Run as a SLURM array (Run_09.sh):
#   tasks 0 to len(TRANSPLANT_GAMMAS) - 1: the network and the unpenalized logistic regression, with
#       tempered class weights (1/class frequency)^gamma, one gamma per task
#   last task: weighted KNN, which tunes its own k, gamma and beta as in 07
# 10_Transplant_Summary.py then picks each model's gamma (or k, gamma, beta) by validation
# balanced accuracy and scores the test set.
#
# Cells, from the full-data prep (03 with no cap):
#   training: the full run's training set, minus a stratified 10% validation part (as in 06)
#   test: the full run's test set, minus every cell that was in the source run's training set,
#         because the source run's tier-1 model chose the genes using those cells
# Genes are centred and scaled to unit variance on the training part (network and baseline); KNN
# scales them by SD^beta.
#
# Needs:    X.npz and prep.npz from 03 on the full data, in <STORAGE_DIR>/<run>/reed_level2_classifier/
#           prep.npz and model_max<G>genes.npz from the source run, in .../reed_level2_classifier_cap<N>/
# Produces: in <STORAGE_DIR>/<run>/reed_level2_transplant/:
#           gamma<g>.npz    per network/baseline task: validation balanced accuracy and macro F1, and the
#                           test-set class probabilities, for both models; nn_gamma<g>.pt (the network)
#                           and nn_training_curve_gamma<g>.csv
#           knn.npz         the KNN tuning grid's validation scores, and the chosen setting's test scores
#           test_cells.csv.gz  the test cells and their states (the same in every task)
#
# Usage: sbatch Run_09.sh [cellranger_run_name]

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

from level2_classifier_common import (OUTPUT_DIR, CLASSES, RANDOM_STATE, NN_VAL_SIZE, NN_HIDDEN, KNN_K, KNN_GAMMAS,
                                      KNN_BETAS, TRANSPLANT_SOURCE_CAP, TRANSPLANT_SOURCE_MAX_GENES,
                                      TRANSPLANT_GAMMAS, get_classifier_dir, get_figure_dir, load_prepared,
                                      get_class_weights, set_threads)
from level2_tier2_models import train_network, network_proba, knn_class_vote_sums, knn_predict

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

cellranger_run_name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else '90'
out_dir = os.path.join(OUTPUT_DIR, cellranger_run_name, 'reed_level2_transplant')
os.makedirs(out_dir, exist_ok=True)
get_figure_dir(cellranger_run_name, None, '09')   # no figures (10 makes them); made like every step
task = int(os.environ['SLURM_ARRAY_TASK_ID'])
n_tasks = len(TRANSPLANT_GAMMAS) + 1
if not 0 <= task < n_tasks:
    raise ValueError(f"array task {task} is out of range: there are {n_tasks} tasks (0-{n_tasks - 1}). "
                     "Check that the --array line in Run_09.sh matches TRANSPLANT_GAMMAS.")
threads = set_threads()
print(f"Cellranger run: {cellranger_run_name}, task {task}, {threads} threads")

# ---------------------- The source run's genes, and the cells it trained on
src_dir = get_classifier_dir(cellranger_run_name, TRANSPLANT_SOURCE_CAP)
with np.load(os.path.join(src_dir, f'model_max{TRANSPLANT_SOURCE_MAX_GENES}genes.npz')) as f:
    selected = np.linalg.norm(f['W'], axis=1) > 0
    src_genes = f['genes']
with np.load(os.path.join(src_dir, 'prep.npz')) as f:
    src_train_cells = set(f['cells'][f['idx_train']])
print(f"{selected.sum()} genes from the {TRANSPLANT_SOURCE_CAP:,}-per-class run: {', '.join(src_genes[selected])}\n")

# ---------------------- Full data: split, test cells, scaling
X, prep = load_prepared(get_classifier_dir(cellranger_run_name, None))
assert (prep['genes'] == src_genes).all(), "gene order differs between the source run and the full-data prep"
X = X[:, np.flatnonzero(selected)].toarray()
y, cells = prep['y'], prep['cells']
idx_fit, idx_val = train_test_split(prep['idx_train'], test_size=NN_VAL_SIZE,
                                    stratify=y[prep['idx_train']], random_state=RANDOM_STATE)
idx_test = np.array([i for i in prep['idx_test'] if cells[i] not in src_train_cells])
log(f"train: {len(idx_fit):,} cells, validation: {len(idx_val):,}, test: {len(idx_test):,} "
    f"({len(prep['idx_test']) - len(idx_test):,} full-run test cells dropped: in the source run's training set)")
if task == 0:
    pd.DataFrame({'cell': cells[idx_test], 'state': y[idx_test]}) \
        .to_csv(os.path.join(out_dir, 'test_cells.csv.gz'), index=False)
y_fit, y_val = y[idx_fit], y[idx_val]


def val_scores(pred):
    return balanced_accuracy_score(y_val, pred), f1_score(y_val, pred, labels=CLASSES, average='macro', zero_division=0)


if task < len(TRANSPLANT_GAMMAS):
    # ---------------------- Network and baseline, with class weights (1/frequency)^gamma
    gamma = TRANSPLANT_GAMMAS[task]
    mean, sd = X[idx_fit].mean(axis=0), X[idx_fit].std(axis=0)
    sd[sd == 0] = 1.0
    X_fit, X_val, X_test = ((X[i] - mean) / sd for i in (idx_fit, idx_val, idx_test))
    class_weights = get_class_weights(y_fit) ** gamma
    print(f"gamma {gamma}: class weights " + ", ".join(f"{c} {w:.3g}" for c, w in zip(CLASSES, class_weights)))

    baseline = LogisticRegression(C=np.inf, max_iter=5000,
                                  class_weight=dict(zip(CLASSES, class_weights))).fit(X_fit, y_fit)
    reorder = pd.Index(baseline.classes_).get_indexer(CLASSES)     # predict_proba's columns are alphabetical
    P_val_base, P_test_base = (baseline.predict_proba(Xs)[:, reorder] for Xs in (X_val, X_test))
    base_bacc, base_f1 = val_scores(CLASSES[P_val_base.argmax(axis=1)])
    log(f"baseline: validation balanced accuracy {base_bacc:.4f}, macro F1 {base_f1:.4f}")

    net, history = train_network(X_fit, y_fit, X_val, y_val, class_weights, NN_HIDDEN, log=log)
    P_val_nn, P_test_nn = network_proba(net, X_val), network_proba(net, X_test)
    nn_bacc, nn_f1 = val_scores(CLASSES[P_val_nn.argmax(axis=1)])
    log(f"network: validation balanced accuracy {nn_bacc:.4f}, macro F1 {nn_f1:.4f}")

    history.to_csv(os.path.join(out_dir, f'nn_training_curve_gamma{gamma}.csv'), index=False)
    torch.save(dict(state_dict=net.state_dict(), hidden=NN_HIDDEN, genes=list(src_genes[selected]),
                    classes=list(CLASSES), scaler_mean=mean, scaler_scale=sd, gamma=gamma),
               os.path.join(out_dir, f'nn_gamma{gamma}.pt'))
    np.savez(os.path.join(out_dir, f'gamma{gamma}.npz'), gamma=gamma,
             base_val_bacc=base_bacc, base_val_f1=base_f1, base_P_test=P_test_base,
             nn_val_bacc=nn_bacc, nn_val_f1=nn_f1, nn_P_test=P_test_nn)

else:
    # ---------------------- Weighted KNN, tuning k, gamma and beta (as in 07)
    ref_class = torch.as_tensor(pd.Index(CLASSES).get_indexer(y_fit), dtype=torch.long)
    ref_freq = np.bincount(ref_class.numpy(), minlength=len(CLASSES)) / len(y_fit)
    mean, sd = X[idx_fit].mean(axis=0), X[idx_fit].std(axis=0)
    sd[sd == 0] = 1.0
    scale = lambda idx, beta: torch.as_tensor((X[idx] - mean) / sd ** beta, dtype=torch.float32)
    rows = []
    for beta in KNN_BETAS:
        sums = knn_class_vote_sums(scale(idx_val, beta), scale(idx_fit, beta), ref_class)
        for k in KNN_K:
            for gamma in KNN_GAMMAS:
                bacc, f1 = val_scores(knn_predict(sums[k], ref_freq, gamma)[1])
                rows.append(dict(beta=beta, k=k, gamma=gamma, balanced_accuracy=bacc, f1_macro=f1))
        log(f"beta {beta}: validation scored")
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning['balanced_accuracy'].idxmax()]
    k, gamma, beta = int(best['k']), float(best['gamma']), float(best['beta'])
    log(f"best on validation: k {k}, gamma {gamma}, beta {beta} (balanced accuracy {best['balanced_accuracy']:.4f})")
    test_sums = knn_class_vote_sums(scale(idx_test, beta), scale(idx_fit, beta), ref_class)
    scores_test, _ = knn_predict(test_sums[k], ref_freq, gamma)
    tuning.to_csv(os.path.join(out_dir, 'knn_tuning.csv'), index=False)
    np.savez(os.path.join(out_dir, 'knn.npz'), k=k, gamma=gamma, beta=beta,
             val_bacc=best['balanced_accuracy'], val_f1=best['f1_macro'], scores_test=scores_test)
log("done")
