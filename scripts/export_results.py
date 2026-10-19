# export_results.py
#
# Writes the headline results of one run to results/ (tracked in git, so they can be cited with the
# repo): the tier-2 network's test confusion matrix, the genes, a summary table and the test donors.
# Run it after 05, 06 and 07 (and run it again whenever they're rerun).
#
# Needs:    from the run, in <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/: coef[_max<G>genes].csv (05),
#           test_predictions_<model>[_max<G>genes].npz (05, 06, 07)
# Produces: in results/:
#           confusion_matrix_network.png and .pdf  the network's test confusion matrix (% of each true state)
#           genes.csv        the genes the tier-1 model selected, by importance (size of their coefficients)
#           summary.csv      each model's test balanced accuracy, with the test donors, cells and genes
#           test_donors.csv  the held-out test donors
#
# Usage: sbatch slurm/Run_export.sh [max_cells_per_class] [max_genes]     (defaults: none 40, the full data)

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from hbca_classifier.settings import get_args, get_classifier_dir
from hbca_classifier.plots import load_predictions, confusion_matrix_figure

max_cells_per_class = get_args()
max_genes = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '' else '40'
suffix = '' if max_genes == 'none' else f'_max{max_genes}genes'
clf_dir = get_classifier_dir(max_cells_per_class)
os.makedirs('results', exist_ok=True)

# Genes, by importance
coef = pd.read_csv(os.path.join(clf_dir, f'coef{suffix}.csv'), index_col=0)
genes = coef.drop(index='(intercept)').query('importance > 0').sort_values('importance', ascending=False)
genes[['importance']].rename_axis('gene').round(4).to_csv('results/genes.csv')
print(f"{len(genes)} genes -> results/genes.csv")

# Each model's test balanced accuracy
names = {'nn': 'Neural network (tier 2)', 'baseline': 'Logistic regression (tier 2 baseline)',
         'knn': 'Weighted KNN (tier 2)', 'tier1': 'Group elastic net (tier 1)'}
rows, nn = [], None
for model in names:
    path = os.path.join(clf_dir, f'test_predictions_{model}{suffix}.npz')
    if not os.path.exists(path):
        continue
    d = load_predictions(path)
    rows.append(dict(model=names[model], balanced_accuracy=round(balanced_accuracy_score(d['y_true'], d['pred']), 4),
                     test_donors=len(np.unique(d['donors'])), test_cells=len(d['y_true']), genes=len(genes)))
    if model == 'nn':
        nn = d
summary = pd.DataFrame(rows)
summary.to_csv('results/summary.csv', index=False)
print(summary.to_string(index=False))

# The network's confusion matrix and the test donors
if nn is None:
    raise FileNotFoundError(f"test_predictions_nn{suffix}.npz not found in {clf_dir}: run 06 first")
confusion_matrix_figure(nn['y_true'], nn['pred'], nn['donors'], nn['label'], 'results/confusion_matrix_network')
pd.Series(sorted(np.unique(nn['donors'])), name='donor_id').to_csv('results/test_donors.csv', index=False)
print("-> results/confusion_matrix_network.png, .pdf and results/test_donors.csv")
