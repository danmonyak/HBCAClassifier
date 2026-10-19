# figures.py
#
# Redraws a run's result figures without refitting anything: the test confusion matrix of every model
# (from its saved test predictions) and the tier-1 CV curves (from cv_summary.csv and each model[_max<G>genes].npz).
# Use it after changing a figure in hbca_classifier/plots.py, or to restyle it. The fitting steps
# (05, 06, 07) draw the same figures when they run.
#
# Needs:    test_predictions_<model><suffix>.npz files (from 05, 06, 07), cv_summary.csv and
#           model[_max<G>genes].npz (05), and prep.npz (03, for the donors of older predictions files), in
#           <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/
# Produces: for each predictions file, its test confusion matrix (.png and .pdf), in the step's folder
#           figures/<step>[/cap<N>]/; for each tier-1 model, figures/05[/cap<N>]/cv_curves[_max<G>genes].png
#
# Usage: sbatch slurm/Run_figures.sh [max_cells_per_class]
#        (python scripts/figures.py [max_cells_per_class]; 'none' or nothing for the full data)

import glob
import os

import numpy as np
import pandas as pd

from hbca_classifier.settings import get_args, get_classifier_dir, get_figure_dir
from hbca_classifier.plots import load_predictions, draw_model_figures, cv_curves

max_cells_per_class = get_args()
clf_dir = get_classifier_dir(max_cells_per_class)
files = sorted(glob.glob(os.path.join(clf_dir, 'test_predictions_*.npz')))
if not files:
    raise FileNotFoundError(f"no test_predictions_*.npz in {clf_dir}: run 05, 06 or 07 first")
for path in files:
    d = load_predictions(path)
    stem = draw_model_figures(path, get_figure_dir(max_cells_per_class, d['step']))
    print(f"{os.path.basename(path)} -> {stem}.png, .pdf")

# Tier-1 CV curves, one per fitted tier-1 model (model.npz, model_max<G>genes.npz)
cv_summary = pd.read_csv(os.path.join(clf_dir, 'cv_summary.csv'))
for path in sorted(glob.glob(os.path.join(clf_dir, 'model*.npz'))):
    suffix = os.path.basename(path)[len('model'):-len('.npz')]
    max_genes = int(suffix[len('_max'):-len('genes')]) if suffix else None
    with np.load(path) as f:
        best_l1_ratio, best_alpha = float(f['l1_ratio']), float(f['alpha'])
    out = os.path.join(get_figure_dir(max_cells_per_class, '05'), f'cv_curves{suffix}.png')
    cv_curves(cv_summary, best_l1_ratio, best_alpha, max_genes, out)
    print(f"{os.path.basename(path)} -> {out}")
