# 05_Level2_Classifier_Fit.py
#
# Step 3 of 3 of the level2 classifier: collects the cross-validation results from 04, picks the
# (l1_ratio, alpha) pair with the lowest mean held-out log-loss, refits it on the whole training
# set, and evaluates it on the test set.
#
# Needs:    from 03 and 04, in <STORAGE_DIR>/<run>/reed_level2_classifier[_cap<N>]/:
#           X.npz, prep.npz, and all 20 cv/fold<k>_l1ratio<r>.csv files
# Produces: in that folder:
#           cv_summary.csv   mean CV results for every l1_ratio and alpha (and their standard errors)
#           coef.csv         the fitted coefficients: intercept first, then one row per gene, one column
#                            per class, plus an importance column (size of the gene's coefficients)
#           test_report.csv  per-class precision, recall and F1 on the test set
#           model.npz        W, b, the genes, classes and the scaler's per-gene scale, to apply the
#                            model to other cells: P = softmax((x / scale) @ W + b)
#           in figures/<run>/05/ (or .../05/cap<N>/): cv_curves.png and test_confusion_matrix.png
#
# Usage: python 05_Level2_Classifier_Fit.py [cellranger_run_name] [max_cells_per_class]

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)

from level2_classifier_common import (CLASSES, N_FOLDS, L1_RATIOS, TOL, MAX_ITER,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared,
                                      one_hot, mean_log_loss, set_threads)
from multinomial_group_enet_torch import MultinomialGroupEnet, predict_proba

warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta.*')

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

cellranger_run_name, max_cells_per_class = get_args()
clf_dir = get_classifier_dir(cellranger_run_name, max_cells_per_class)
figure_outdir_cur = get_figure_dir(cellranger_run_name, max_cells_per_class, '05')
threads = set_threads()
print(f"Cellranger run: {cellranger_run_name}, max cells per class: {max_cells_per_class}, {threads} threads\n")

# ---------------------- Collect the CV results
cv_files = [os.path.join(clf_dir, 'cv', f'fold{k}_l1ratio{l1r}.csv') for k in range(N_FOLDS) for l1r in L1_RATIOS]
missing = [f for f in cv_files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"{len(missing)} of {len(cv_files)} CV results are missing (did a 04 task fail?):\n"
                            + "\n".join(missing))
cv_results = pd.concat([pd.read_csv(f) for f in cv_files], ignore_index=True)
if not cv_results['converged'].all():
    print(f"**Warning:** {(~cv_results['converged']).sum()} CV fits didn't converge within {MAX_ITER} iterations\n")

cv_summary = (cv_results.groupby(['l1_ratio', 'alpha'])
              .agg(log_loss=('log_loss', 'mean'), log_loss_se=('log_loss', 'sem'),
                   accuracy=('accuracy', 'mean'), balanced_accuracy=('balanced_accuracy', 'mean'),
                   n_genes=('n_genes', 'mean'))
              .reset_index())
cv_summary.to_csv(os.path.join(clf_dir, 'cv_summary.csv'), index=False)

best = cv_summary.loc[cv_summary['log_loss'].idxmin()]
best_l1_ratio, best_alpha = best['l1_ratio'], best['alpha']
print(f"Best: l1_ratio = {best_l1_ratio}, alpha = {best_alpha:.3g} "
      f"(mean CV log-loss {best['log_loss']:.4f}, balanced accuracy {best['balanced_accuracy']:.3f}, "
      f"~{best['n_genes']:.0f} genes)\n")

# CV curves: mean held-out log-loss (+/- 1 standard error) against alpha, one line per l1_ratio
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for l1r, d in cv_summary.groupby('l1_ratio'):
    axes[0].errorbar(d['alpha'], d['log_loss'], yerr=d['log_loss_se'], marker='o', ms=3, capsize=2, label=f'l1_ratio = {l1r}')
    axes[1].plot(d['alpha'], d['n_genes'], marker='o', ms=3, label=f'l1_ratio = {l1r}')
axes[0].axvline(best_alpha, color='grey', ls='--', lw=1)
axes[0].set(xscale='log', xlabel='alpha', ylabel='Mean CV log-loss', title='Cross-validation')
axes[1].set(xscale='log', xlabel='alpha', ylabel='Genes kept (mean over folds)', title='Model size')
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, 'cv_curves.png'), dpi=200)

# ---------------------- Refit on the whole training set
X, prep = load_prepared(clf_dir)
genes, y = prep['genes'], prep['y']
idx_train, idx_test = prep['idx_train'], prep['idx_test']
scaler = StandardScaler(with_mean=False).fit(X[idx_train])
X_train, X_test = scaler.transform(X[idx_train]), scaler.transform(X[idx_test])
y_train, y_test = y[idx_train], y[idx_test]
del X

model = MultinomialGroupEnet(X_train, one_hot(y_train))
log(f"refitting on {len(idx_train):,} training cells")

# Warm-start along the grid down to the chosen alpha (faster than fitting it from scratch)
grid = prep['alpha_grids'][L1_RATIOS.index(best_l1_ratio)]
W, b = None, None
for alpha in grid[grid >= best_alpha * (1 - 1e-9)]:
    W, b, info = model.fit(alpha, best_l1_ratio, W, b, tol=TOL, max_iter=MAX_ITER)
    log(f"alpha {alpha:.3g}: {info['n_iter']} iterations{'' if info['converged'] else ' (NOT converged)'}")

# ---------------------- Evaluate on the test set
P_test = predict_proba(X_test, W, b)
pred_test = CLASSES[P_test.argmax(axis=1)]
print()
print(f"Test log-loss:          {mean_log_loss(y_test, P_test):.4f}")
print(f"Test accuracy:          {accuracy_score(y_test, pred_test):.4f}")
print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, pred_test):.4f}")
print()
print(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0))
pd.DataFrame(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0, output_dict=True)) \
    .T.to_csv(os.path.join(clf_dir, 'test_report.csv'))

# Confusion matrix on the test set, as the fraction of each true class (rows sum to 1)
cm = confusion_matrix(y_test, pred_test, labels=CLASSES, normalize='true')
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
            vmin=0, vmax=1, cbar_kws={'label': 'Fraction of true class'}, ax=ax)
ax.set(xlabel='Predicted', ylabel='True', title='Test set confusion matrix')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, 'test_confusion_matrix.png'), dpi=200)

# ---------------------- Coefficients
# On the scaled-gene scale. Genes dropped by the group lasso have all-zero rows.
coef = pd.DataFrame(np.vstack([b, W]), index=np.r_[['(intercept)'], genes], columns=CLASSES)
coef['importance'] = np.r_[np.nan, np.linalg.norm(W, axis=1)]
coef.to_csv(os.path.join(clf_dir, 'coef.csv'))
np.savez(os.path.join(clf_dir, 'model.npz'), W=W, b=b, genes=genes, classes=CLASSES, scale=scaler.scale_,
         l1_ratio=best_l1_ratio, alpha=best_alpha)

selected = coef.drop(index='(intercept)').query('importance > 0').sort_values('importance', ascending=False)
print(f"{len(selected):,} of {len(genes):,} genes kept. Top 30:")
print(selected.head(30).round(3).to_string())
log("done")
