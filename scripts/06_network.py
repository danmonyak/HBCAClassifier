# 06_network.py
#
# Predicts the level2 class (10 epithelial states + "Other") from the small gene set that the
# group-lasso model in 05 selected, with a dense neural network. As a baseline, it also fits an
# unpenalized multinomial logistic regression on the same genes: if the network doesn't beat it,
# the nonlinearity isn't adding anything.
#
# Both models use the same cells: the training set from 03, minus a validation part (the cells of
# about NN_VAL_SIZE of the training donors; donor_split), which the network uses for early stopping. Both weight each class by its inverse
# frequency, so every class counts equally (as in balanced accuracy and macro F1). The test set is
# used once, at the end. Settings are in hbca_classifier/settings.py.
#
# Needs:    from 03 and 05 (run with the same arguments), in
#           <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/: X.npz, prep.npz, model[_max<G>genes].npz
# Produces: in that folder (all with the _max<G>genes suffix when max_genes is given):
#           nn.pt                 the network's weights, plus the genes, classes and input scaling
#           nn_comparison.csv     test log-loss, accuracy, balanced accuracy and macro F1, for both models
#           nn_test_report.csv, baseline_test_report.csv   per-class precision, recall and F1 on the test set
#           in figures/06/ (or .../06/cap<N>/): nn_training_curve.png (training and validation
#           loss per epoch), nn_test_confusion_matrix.png and baseline_test_confusion_matrix.png
#
# Optional 3rd argument: the name of another network shape in NN_ARCHITECTURES (e.g. deep). Every
# output then gets an extra _<name> suffix, so the default network's results are kept.
#
# Usage: python scripts/06_network.py [max_cells_per_class] [max_genes] [architecture]
#        e.g. after `sbatch slurm/Run_05.sh 300 40`: sbatch slurm/Run_06.sh 300 40

import copy
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
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from hbca_classifier.settings import (score_text, donor_split, describe_split, CLASSES, RANDOM_STATE, NN_VAL_SIZE, NN_HIDDEN, NN_ARCHITECTURES, NN_DROPOUT,
                                      NN_LEARNING_RATE, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_PATIENCE,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared,
                                      mean_log_loss, set_threads, get_class_weights)

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

max_cells_per_class = get_args()
max_genes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ('', 'none') else None
suffix = '' if max_genes is None else f'_max{max_genes}genes'     # matches 05's output names
architecture = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ('', 'none', 'default') else None
if architecture is not None and architecture not in NN_ARCHITECTURES:
    raise ValueError(f"unknown architecture {architecture!r}: choose from {list(NN_ARCHITECTURES)}")
hidden = NN_HIDDEN if architecture is None else NN_ARCHITECTURES[architecture]
out = suffix + ('' if architecture is None else f'_{architecture}')   # this run's output file suffix
clf_dir = get_classifier_dir(max_cells_per_class)
figure_outdir_cur = get_figure_dir(max_cells_per_class, '06')
threads = set_threads()
torch.manual_seed(RANDOM_STATE)
print(f"Max cells per class: {max_cells_per_class}, "
      f"max genes: {max_genes}, {threads} threads\n")

# ---------------------- The genes selected by 05
model_file = os.path.join(clf_dir, f'model{suffix}.npz')
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found: run 05 with the same arguments first")
with np.load(model_file) as f:
    selected = np.linalg.norm(f['W'], axis=1) > 0
    genes = f['genes'][selected]
print(f"{len(genes)} genes selected by 05: {', '.join(genes)}\n")

# ---------------------- Data: the selected genes only (small, so dense)
X, prep = load_prepared(clf_dir)
X = X[:, np.flatnonzero(selected)].toarray()
y = prep['y']
idx_test = prep['idx_test']
# Split the training set into the part both models train on and the network's validation part:
# the cells of about NN_VAL_SIZE of the training donors
idx_train = prep['idx_train']
is_val, _ = donor_split(prep['donors'][idx_train], y[idx_train], NN_VAL_SIZE)
idx_fit, idx_val = idx_train[~is_val], idx_train[is_val]
describe_split(y[idx_train], is_val, prep['donors'][idx_train], 'Validation')
# Centre and scale each gene on the training part only
scaler = StandardScaler().fit(X[idx_fit])
X_fit, X_val, X_test = (scaler.transform(X[i]) for i in (idx_fit, idx_val, idx_test))
y_fit, y_val, y_test = y[idx_fit], y[idx_val], y[idx_test]
log(f"train: {len(idx_fit):,} cells, validation: {len(idx_val):,}, test: {len(idx_test):,}")

# Class weights: inverse frequency in the training part, scaled so the average cell has weight 1
class_weights = get_class_weights(y_fit)


def evaluate(name, P):
    """Test-set metrics for class probabilities P (columns in the order of CLASSES)."""
    pred = CLASSES[P.argmax(axis=1)]
    print(f"\n{name} on the test set:")
    print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))
    return pred, dict(model=name, log_loss=mean_log_loss(y_test, P, class_weights),
                      accuracy=accuracy_score(y_test, pred),
                      balanced_accuracy=balanced_accuracy_score(y_test, pred),
                      f1_macro=f1_score(y_test, pred, labels=CLASSES, average='macro', zero_division=0))


def save_confusion_matrix(pred, title, file_name):
    """Confusion matrix on the test set, as the fraction of each true class (rows sum to 1)."""
    cm = confusion_matrix(y_test, pred, labels=CLASSES, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
                vmin=0, vmax=1, cbar_kws={'label': 'Fraction of true class'}, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title=f'{title}\n{score_text(y_test, pred)}')
    fig.tight_layout()
    fig.savefig(os.path.join(figure_outdir_cur, file_name), dpi=200)
    plt.close(fig)


def save_report(pred, file_name):
    pd.DataFrame(classification_report(y_test, pred, labels=CLASSES, zero_division=0, output_dict=True)) \
        .T.to_csv(os.path.join(clf_dir, file_name))


# ---------------------- Baseline: unpenalized multinomial logistic regression
# C = inf means no penalty. Classes are weighted the same way as in the network.
baseline = LogisticRegression(C=np.inf, max_iter=5000,
                              class_weight=dict(zip(CLASSES, class_weights))).fit(X_fit, y_fit)
log(f"baseline fitted ({baseline.n_iter_[0]} iterations)")
# predict_proba's columns follow baseline.classes_ (alphabetical): reorder them to CLASSES
P_base = baseline.predict_proba(X_test)[:, pd.Index(baseline.classes_).get_indexer(CLASSES)]
pred_base, metrics_base = evaluate('Logistic regression (unpenalized)', P_base)
save_report(pred_base, f'baseline_test_report{out}.csv')
save_confusion_matrix(pred_base, f'Logistic regression, {len(genes)} genes: test set',
                      f'baseline_test_confusion_matrix{out}.png')

# ---------------------- Neural network
# Dense layers with ReLU and dropout, then one output per class (softmax is inside the loss)
layers, n_in = [], len(genes)
for n_out in hidden:
    layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(NN_DROPOUT)]
    n_in = n_out
layers.append(nn.Linear(n_in, len(CLASSES)))
net = nn.Sequential(*layers)
print(f"\n{net}\n")

to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
to_index = lambda labels: torch.as_tensor(pd.Index(CLASSES).get_indexer(labels), dtype=torch.long)
Xt_fit, yt_fit = to_tensor(X_fit), to_index(y_fit)
Xt_val, yt_val = to_tensor(X_val), to_index(y_val)

loss_fn = nn.CrossEntropyLoss(weight=to_tensor(class_weights))
optimizer = torch.optim.Adam(net.parameters(), lr=NN_LEARNING_RATE)
history, best_val, best_state, best_epoch = [], np.inf, None, 0
for epoch in range(1, NN_MAX_EPOCHS + 1):
    net.train()
    order = torch.randperm(len(yt_fit))
    total = 0.
    for start in range(0, len(order), NN_BATCH_SIZE):
        batch = order[start:start + NN_BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_fn(net(Xt_fit[batch]), yt_fit[batch])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    net.eval()
    with torch.no_grad():
        val_loss = loss_fn(net(Xt_val), yt_val).item()
    history.append(dict(epoch=epoch, train_loss=total / len(yt_fit), val_loss=val_loss))
    # Early stopping: keep the weights with the lowest validation loss
    if val_loss < best_val:
        best_val, best_state, best_epoch = val_loss, copy.deepcopy(net.state_dict()), epoch
    elif epoch - best_epoch >= NN_PATIENCE:
        break
    if epoch % 10 == 0:
        log(f"epoch {epoch}: training loss {history[-1]['train_loss']:.4f}, validation loss {val_loss:.4f}")
net.load_state_dict(best_state)
log(f"stopped after {epoch} epochs; best validation loss {best_val:.4f} at epoch {best_epoch}")

with torch.no_grad():
    P_nn = torch.softmax(net(to_tensor(X_test)), dim=1).numpy().astype(np.float64)
nn_name = 'Neural network' + ('' if architecture is None else f", {architecture} ({'-'.join(map(str, hidden))})")
pred_nn, metrics_nn = evaluate(nn_name, P_nn)
save_report(pred_nn, f'nn_test_report{out}.csv')
save_confusion_matrix(pred_nn, f'Neural network, {len(genes)} genes: test set',
                      f'nn_test_confusion_matrix{out}.png')

# Training curve: (weighted) loss per epoch, with the epoch kept marked
hist = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(hist['epoch'], hist['train_loss'], label='training')
ax.plot(hist['epoch'], hist['val_loss'], label='validation')
ax.axvline(best_epoch, color='grey', ls='--', lw=1, label=f'kept (epoch {best_epoch})')
ax.set(xlabel='Epoch', ylabel='Class-weighted cross-entropy', title=f'Neural network, {len(genes)} genes\ntest set: {score_text(y_test, pred_nn)}')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'nn_training_curve{out}.png'), dpi=200)
plt.close(fig)

# ---------------------- Compare and save
comparison = pd.DataFrame([metrics_base, metrics_nn])
comparison.to_csv(os.path.join(clf_dir, f'nn_comparison{out}.csv'), index=False)
print("\nTest set (log-loss is class-weighted):")
print(comparison.round(4).to_string(index=False))

torch.save(dict(state_dict=net.state_dict(), hidden=hidden, dropout=NN_DROPOUT, genes=list(genes),
                classes=list(CLASSES), scaler_mean=scaler.mean_, scaler_scale=scaler.scale_),
           os.path.join(clf_dir, f'nn{out}.pt'))
log("done")
