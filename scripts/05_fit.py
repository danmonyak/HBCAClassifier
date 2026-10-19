# 05_fit.py
#
# Step 3 of 3 of the level2 classifier: collects the cross-validation results from 04, picks the
# (l1_ratio, alpha) pair with the lowest mean held-out log-loss, refits it on the whole training
# set, and evaluates it on the test set. Cells are weighted by inverse class frequency, as in 04.
#
# Optional max_genes (second argument): pick the best pair among those that keep at most that many
# genes (mean over the CV folds) instead. Its outputs get a _max<N>genes suffix, so they don't
# overwrite the unconstrained ones. The refitted model can keep a few more or fewer genes than
# the CV mean.
#
# Needs:    from 03 and 04, in <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/:
#           X.npz, prep.npz, and all 20 cv/fold<k>_l1ratio<r>.csv files
# Produces: in that folder:
#           cv_summary.csv   mean CV results for every l1_ratio and alpha (and their standard errors)
#           coef.csv         the fitted coefficients: intercept first, then one row per gene, one column
#                            per class, plus an importance column (size of the gene's coefficients)
#           test_report.csv  per-class precision, recall and F1 on the test set
#           model.npz        W, b, the genes, classes and the scaler's per-gene scale, to apply the
#                            model to other cells: P = softmax((x / scale) @ W + b)
#           in figures/05/ (or .../05/cap<N>/): cv_curves.png (log-loss, macro F1 and genes
#           kept against alpha) and test_confusion_matrix.png
#           With max_genes, every file above except cv_summary.csv gets a _max<N>genes suffix.
#
# Usage: python scripts/05_fit.py [max_cells_per_class] [max_genes]
#        e.g. the full data, at most 200 genes: sbatch slurm/Run_05.sh none 200

import os
import sys
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
                             classification_report, confusion_matrix, f1_score)

from hbca_classifier.settings import (score_text, CLASSES, N_FOLDS, L1_RATIOS, TOL, MAX_ITER,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared,
                                      one_hot, mean_log_loss, set_threads,
                                      get_class_weights, get_sample_weights, get_scaling_weights)
from hbca_classifier.group_enet import MultinomialGroupEnet, predict_proba

warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta.*')

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

max_cells_per_class = get_args()
clf_dir = get_classifier_dir(max_cells_per_class)
figure_outdir_cur = get_figure_dir(max_cells_per_class, '05')
threads = set_threads()
max_genes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ('', 'none') else None
suffix = '' if max_genes is None else f'_max{max_genes}genes'     # added to every output file name
print(f"Max cells per class: {max_cells_per_class}, "
      f"max genes: {max_genes}, {threads} threads\n")

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
                   f1_macro=('f1_macro', 'mean'), f1_macro_se=('f1_macro', 'sem'),
                   n_genes=('n_genes', 'mean'))
              .reset_index())
cv_summary.to_csv(os.path.join(clf_dir, 'cv_summary.csv'), index=False)

# The best pair: lowest mean CV log-loss, among the pairs with at most max_genes genes if it's set
eligible = cv_summary if max_genes is None else cv_summary[cv_summary['n_genes'] <= max_genes]
if eligible.empty:
    raise ValueError(f"No (l1_ratio, alpha) pair keeps at most {max_genes} genes. "
                     f"The smallest models keep {cv_summary['n_genes'].min():.0f}.")
best = eligible.loc[eligible['log_loss'].idxmin()]
best_l1_ratio, best_alpha = best['l1_ratio'], best['alpha']
print(f"Best{'' if max_genes is None else f' with at most {max_genes} genes'}: "
      f"l1_ratio = {best_l1_ratio}, alpha = {best_alpha:.3g} "
      f"(mean CV log-loss {best['log_loss']:.4f}, balanced accuracy {best['balanced_accuracy']:.3f}, "
      f"macro F1 {best['f1_macro']:.3f}, ~{best['n_genes']:.0f} genes)\n")

# CV curves against alpha, one line per l1_ratio: mean held-out log-loss and macro F1 (+/- 1 standard
# error), and genes kept. The black star marks the chosen pair.
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for l1r, d in cv_summary.groupby('l1_ratio'):
    axes[0].errorbar(d['alpha'], d['log_loss'], yerr=d['log_loss_se'], marker='o', ms=3, capsize=2, label=f'l1_ratio = {l1r}')
    axes[1].errorbar(d['alpha'], d['f1_macro'], yerr=d['f1_macro_se'], marker='o', ms=3, capsize=2, label=f'l1_ratio = {l1r}')
    axes[2].plot(d['alpha'], d['n_genes'], marker='o', ms=3, label=f'l1_ratio = {l1r}')
for ax, col in zip(axes, ['log_loss', 'f1_macro', 'n_genes']):
    ax.plot(best_alpha, best[col], marker='*', ms=14, color='black', ls='none', zorder=5)
if max_genes is not None:
    axes[2].axhline(max_genes, color='grey', ls='--', lw=1)
axes[0].set(xscale='log', xlabel='alpha', ylabel='Mean CV log-loss', title='Cross-validation: log-loss')
axes[1].set(xscale='log', xlabel='alpha', ylabel='Mean CV macro F1', title='Cross-validation: macro F1')
axes[2].set(xscale='log', xlabel='alpha', ylabel='Genes kept (mean over folds)', title='Model size')
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'cv_curves{suffix}.png'), dpi=200)

# ---------------------- Refit on the whole training set
X, prep = load_prepared(clf_dir)
genes, y = prep['genes'], prep['y']
idx_train, idx_test = prep['idx_train'], prep['idx_test']
y_train, y_test = y[idx_train], y[idx_test]
scaler = StandardScaler(with_mean=False).fit(X[idx_train], sample_weight=get_scaling_weights(y_train))
X_train, X_test = scaler.transform(X[idx_train]), scaler.transform(X[idx_test])
del X

class_weights = get_class_weights(y_train)     # inverse class frequency, as in 04
model = MultinomialGroupEnet(X_train, one_hot(y_train), sample_weight=get_sample_weights(y_train, class_weights))
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
print(f"Test log-loss:          {mean_log_loss(y_test, P_test, class_weights):.4f} (class-weighted)")
print(f"Test accuracy:          {accuracy_score(y_test, pred_test):.4f}")
print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, pred_test):.4f}")
print(f"Test macro F1:          {f1_score(y_test, pred_test, labels=CLASSES, average='macro', zero_division=0):.4f}")
print()
print(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0))
pd.DataFrame(classification_report(y_test, pred_test, labels=CLASSES, zero_division=0, output_dict=True)) \
    .T.to_csv(os.path.join(clf_dir, f'test_report{suffix}.csv'))

# Confusion matrix on the test set, as the fraction of each true class (rows sum to 1)
cm = confusion_matrix(y_test, pred_test, labels=CLASSES, normalize='true')
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
            vmin=0, vmax=1, cbar_kws={'label': 'Fraction of true class'}, ax=ax)
ax.set(xlabel='Predicted', ylabel='True',
       title=f'Test set confusion matrix ({(np.linalg.norm(W, axis=1) > 0).sum()} genes, '
             f'l1_ratio {best_l1_ratio}, alpha {best_alpha:.3g})\n{score_text(y_test, pred_test)}')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'test_confusion_matrix{suffix}.png'), dpi=200)

# ---------------------- Coefficients
# On the scaled-gene scale. Genes dropped by the group lasso have all-zero rows.
coef = pd.DataFrame(np.vstack([b, W]), index=np.r_[['(intercept)'], genes], columns=CLASSES)
coef['importance'] = np.r_[np.nan, np.linalg.norm(W, axis=1)]
coef.to_csv(os.path.join(clf_dir, f'coef{suffix}.csv'))
np.savez(os.path.join(clf_dir, f'model{suffix}.npz'), W=W, b=b, genes=genes, classes=CLASSES, scale=scaler.scale_,
         l1_ratio=best_l1_ratio, alpha=best_alpha)

selected = coef.drop(index='(intercept)').query('importance > 0').sort_values('importance', ascending=False)
print(f"{len(selected):,} of {len(genes):,} genes kept. Top 30:")
print(selected.head(30).round(3).to_string())
log("done")
