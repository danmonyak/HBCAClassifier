# 10_transplant_summary.py
#
# TRANSPLANT, step 2 of 2: collects 09's results, picks the class-weight exponent gamma for the
# network and for the baseline by validation balanced accuracy (KNN chose its own k, gamma and beta
# in 09), and scores the chosen models once on the test set.
#
# Needs:    from 09, in <STORAGE_DIR>/reed_level2_transplant/: gamma<g>.npz for every
#           TRANSPLANT_GAMMAS value, knn.npz and test_cells.csv.gz
# Produces: in that folder:
#           gamma_tuning.csv          validation balanced accuracy and macro F1 per gamma, network and baseline
#           transplant_comparison.csv test log-loss (class-weighted), accuracy, balanced accuracy and macro F1
#                                     for the three chosen models
#           <model>_test_report.csv   per-class precision, recall and F1 (model: nn, baseline, knn)
#           in figures/10/: gamma_tuning.png and <model>_test_confusion_matrix.png
#
# Usage: sbatch slurm/Run_10.sh

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from hbca_classifier.settings import (OUTPUT_DIR, CLASSES, TRANSPLANT_GAMMAS, TRANSPLANT_SOURCE_CAP,
                                      get_figure_dir, get_class_weights, mean_log_loss)

out_dir = os.path.join(OUTPUT_DIR, 'reed_level2_transplant')
figure_outdir_cur = get_figure_dir(None, '10')

files = [os.path.join(out_dir, f'gamma{g}.npz') for g in TRANSPLANT_GAMMAS] + [os.path.join(out_dir, 'knn.npz')]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError("missing results from 09 (did a task fail?):\n" + "\n".join(missing))
y_test = pd.read_csv(os.path.join(out_dir, 'test_cells.csv.gz'))['state'].to_numpy()
test_weights = get_class_weights(y_test)     # for a log-loss in which every class counts equally
print(f"Test cells: {len(y_test):,}\n")

# ---------------------- Pick gamma for the network and the baseline
runs = {g: dict(np.load(os.path.join(out_dir, f'gamma{g}.npz'))) for g in TRANSPLANT_GAMMAS}
tuning = pd.DataFrame([dict(gamma=g, nn_val_bacc=float(r['nn_val_bacc']), nn_val_f1=float(r['nn_val_f1']),
                            base_val_bacc=float(r['base_val_bacc']), base_val_f1=float(r['base_val_f1']))
                       for g, r in runs.items()])
tuning.to_csv(os.path.join(out_dir, 'gamma_tuning.csv'), index=False)
print("Validation scores per gamma:")
print(tuning.round(4).to_string(index=False))
g_nn = tuning.loc[tuning['nn_val_bacc'].idxmax(), 'gamma']
g_base = tuning.loc[tuning['base_val_bacc'].idxmax(), 'gamma']
knn = dict(np.load(os.path.join(out_dir, 'knn.npz')))
print(f"\nChosen: network gamma {g_nn}, baseline gamma {g_base}, "
      f"KNN k {int(knn['k'])} gamma {float(knn['gamma'])} beta {float(knn['beta'])}\n")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(tuning['gamma'], tuning['nn_val_bacc'], marker='o', label='Neural network')
ax.plot(tuning['gamma'], tuning['base_val_bacc'], marker='o', label='Logistic regression')
ax.axhline(float(knn['val_bacc']), color='grey', ls='--', lw=1, label=f"Weighted KNN (best, gamma {float(knn['gamma'])})")
ax.set(xlabel='gamma (class weights = (1 / class frequency)^gamma)', ylabel='Validation balanced accuracy',
       title='TRANSPLANT: class-weight tuning')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, 'gamma_tuning.png'), dpi=200)
plt.close(fig)

# ---------------------- Score the chosen models on the test set
models = {
    'nn': (f'Neural network (gamma {g_nn})', runs[g_nn]['nn_P_test']),
    'baseline': (f'Logistic regression (gamma {g_base})', runs[g_base]['base_P_test']),
    'knn': (f"Weighted KNN (k {int(knn['k'])}, gamma {float(knn['gamma'])}, beta {float(knn['beta'])})", knn['scores_test']),
}
rows = []
for key, (name, P) in models.items():
    pred = CLASSES[P.argmax(axis=1)]
    print(f"{name} on the test set:")
    print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))
    pd.DataFrame(classification_report(y_test, pred, labels=CLASSES, zero_division=0, output_dict=True)) \
        .T.to_csv(os.path.join(out_dir, f'{key}_test_report.csv'))
    rows.append(dict(model=name,
                     # KNN's scores are vote shares (steps of about 1/k), so its log-loss isn't meaningful
                     log_loss=np.nan if key == 'knn' else mean_log_loss(y_test, P, test_weights),
                     accuracy=accuracy_score(y_test, pred),
                     balanced_accuracy=balanced_accuracy_score(y_test, pred),
                     f1_macro=f1_score(y_test, pred, labels=CLASSES, average='macro', zero_division=0)))
    cm = confusion_matrix(y_test, pred, labels=CLASSES, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
                vmin=0, vmax=1, cbar_kws={'label': 'Fraction of true class'}, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title=f'TRANSPLANT, {name}: test set')
    fig.tight_layout()
    fig.savefig(os.path.join(figure_outdir_cur, f'{key}_test_confusion_matrix.png'), dpi=200)
    plt.close(fig)

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(out_dir, 'transplant_comparison.csv'), index=False)
print(f"Test set ({len(y_test):,} cells; genes from the {TRANSPLANT_SOURCE_CAP:,}-per-class run; "
      "log-loss is class-weighted):")
print(comparison.round(4).to_string(index=False))
