# figures.py
#
# Redraws the test figures of every model in a run from their saved test predictions, without
# refitting anything. Use it after changing the figures in hbca_classifier/plots.py, or to restyle
# them. The fitting steps (05, 06, 07) draw the same figures when they run.
#
# Needs:    test_predictions_<model><suffix>.npz files (from 05, 06, 07), in
#           <STORAGE_DIR>/reed_level2_classifier[_cap<N>]/
# Produces: for each file, its three test figures (confusion matrix by true class, by predicted
#           class, and precision/recall per class), in the step's folder: figures/<step>[/cap<N>]/
#
# Usage: sbatch slurm/Run_figures.sh [max_cells_per_class]
#        (python scripts/figures.py [max_cells_per_class]; 'none' or nothing for the full data)

import glob
import os

from hbca_classifier.settings import get_args, get_classifier_dir, get_figure_dir
from hbca_classifier.plots import load_predictions, draw_model_figures

max_cells_per_class = get_args()
clf_dir = get_classifier_dir(max_cells_per_class)
files = sorted(glob.glob(os.path.join(clf_dir, 'test_predictions_*.npz')))
if not files:
    raise FileNotFoundError(f"no test_predictions_*.npz in {clf_dir}: run 05, 06 or 07 first")
for path in files:
    d = load_predictions(path)
    for fig in draw_model_figures(path, get_figure_dir(max_cells_per_class, d['step'])):
        print(f"{os.path.basename(path)} -> {fig}")
