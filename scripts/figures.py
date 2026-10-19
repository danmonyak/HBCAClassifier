# figures.py
#
# Redraws the test confusion matrix of every model in a run from their saved test predictions, without
# refitting anything. Use it after changing the figure in hbca_classifier/plots.py, or to restyle it.
# The fitting steps (05, 06, 07) draw the same figure when they run.
#
# Needs:    test_predictions_<model><suffix>.npz files (from 05, 06, 07), and prep.npz (from 03, for the
#           test cells' donors in files saved before predictions held them), in
#           <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/
# Produces: for each file, its test confusion matrix (.png and .pdf), in the step's folder:
#           figures/<step>[/cap<N>]/
#
# Usage: sbatch slurm/Run_figures.sh [max_cells_per_class]
#        (python scripts/figures.py [max_cells_per_class]; 'none' or nothing for the full data)

import glob
import os

import numpy as np
import pandas as pd

from hbca_classifier.settings import get_args, get_classifier_dir, get_figure_dir
from hbca_classifier.plots import load_predictions, draw_model_figures

max_cells_per_class = get_args()
clf_dir = get_classifier_dir(max_cells_per_class)
files = sorted(glob.glob(os.path.join(clf_dir, 'test_predictions_*.npz')))
if not files:
    raise FileNotFoundError(f"no test_predictions_*.npz in {clf_dir}: run 05, 06 or 07 first")
with np.load(os.path.join(clf_dir, 'prep.npz')) as f:
    donor_of = pd.Series(f['donors'], index=f['cells'])
for path in files:
    d = load_predictions(path)
    stem = draw_model_figures(path, get_figure_dir(max_cells_per_class, d['step']),
                              donors=donor_of.loc[d['cells']].to_numpy())
    print(f"{os.path.basename(path)} -> {stem}.png, .pdf")
