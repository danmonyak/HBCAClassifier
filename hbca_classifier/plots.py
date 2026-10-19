# hbca_classifier/plots.py
#
# The test figure of a model, drawn only from its saved test predictions, so it can be redrawn (and
# restyled) at any time without refitting. The fitting steps (05, 06, 07) save each model's predictions
# with save_predictions and draw its figure with draw_model_figures; scripts/figures.py redraws every
# model of a run from the saved files.
#
# The figure is the test-set confusion matrix as the percentage of each true state (rows sum to 100%).
# Its diagonal is each state's recall, and their mean is the balanced accuracy shown in the title.
#
# A predictions file, test_predictions_<model><suffix>.npz in the run's classifier folder, holds:
#   cells, y_true, donors  the test cells, their true class and their donor
#   P                  the model's class scores (n_cells x 11, columns in the order of CLASSES):
#                      probabilities, or vote shares for KNN; the prediction is the top class
#   label              the model's name for the figure title, e.g. 'Neural network, 35 genes'
#   step, name, suffix where its figure goes: figures/<step>[/cap<N>]/<name>test_confusion_matrix<suffix>.png/.pdf

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

from hbca_classifier.settings import CLASSES


def save_predictions(clf_dir, model, suffix, cells, y_true, donors, P, label, step, name):
    """Save a model's test predictions as <clf_dir>/test_predictions_<model><suffix>.npz; returns the path."""
    path = os.path.join(clf_dir, f'test_predictions_{model}{suffix}.npz')
    np.savez_compressed(path, cells=np.asarray(cells, dtype=str), y_true=np.asarray(y_true, dtype=str),
                        donors=np.asarray(donors, dtype=str), P=np.asarray(P, dtype=np.float32),
                        classes=CLASSES, label=label, step=step, name=name, suffix=suffix)
    return path


def load_predictions(path):
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    assert list(d['classes']) == list(CLASSES), f"{path}: class order differs from CLASSES"
    for k in ('label', 'step', 'name', 'suffix'):
        d[k] = str(d[k])
    d['pred'] = CLASSES[d['P'].argmax(axis=1)]
    return d


def confusion_matrix_figure(y_true, pred, donors, label, path_stem):
    """Confusion matrix as the percentage of each true state (rows sum to 100%), saved as <path_stem>.png
    and .pdf. Cells under 0.5% are left blank."""
    cm = 100 * pd.crosstab(pd.Series(y_true), pd.Series(pred), normalize='index') \
        .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    annot = cm.round().astype(int).astype(str).where(cm >= 0.5, '')
    bacc = balanced_accuracy_score(y_true, pred)
    test = f"{len(np.unique(donors))} held-out donors, " if donors is not None else ''
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=100, linewidths=0.5, linecolor='white',
                annot_kws={'size': 10}, cbar_kws={'label': '% of true state'}, ax=ax)
    ax.set_xlabel('Predicted state', fontsize=12)
    ax.set_ylabel('True state', fontsize=12)
    ax.tick_params(axis='both', labelsize=10, length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_title(f"{label}\nTest set: {test}{len(y_true):,} cells. Balanced accuracy {bacc:.3f}", fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_stem}.{ext}', dpi=300)
    plt.close(fig)


def draw_model_figures(pred_path, fig_dir, donors=None):
    """Draw a model's test confusion matrix from its predictions file into fig_dir; returns the path stem.
    donors: the test cells' donors, for files saved before predictions held them."""
    d = load_predictions(pred_path)
    if 'donors' in d:
        donors = d['donors']
    stem = os.path.join(fig_dir, f"{d['name']}test_confusion_matrix{d['suffix']}")
    confusion_matrix_figure(d['y_true'], d['pred'], donors, d['label'], stem)
    return stem
