# hbca_classifier/plots.py
#
# Figures of a model's test results, drawn only from its saved test predictions, so they can be
# redrawn (and restyled) at any time without refitting. The fitting steps (05, 06, 07) save each
# model's predictions with save_predictions and draw its figures with draw_model_figures;
# scripts/figures.py redraws every model of a run from the saved files.
#
# A predictions file, test_predictions_<model><suffix>.npz in the run's classifier folder, holds:
#   cells, y_true      the test cells and their true class
#   P                  the model's class scores (n_cells x 11, columns in the order of CLASSES):
#                      probabilities, or vote shares for KNN; the prediction is the top class
#   label              the model's name for figure titles, e.g. 'Neural network, 38 genes'
#   step, name, suffix where its figures go: figures/<step>[/cap<N>]/<name>test_...<suffix>.png

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from hbca_classifier.settings import CLASSES, score_text


def save_predictions(clf_dir, model, suffix, cells, y_true, P, label, step, name):
    """Save a model's test predictions as <clf_dir>/test_predictions_<model><suffix>.npz; returns the path."""
    path = os.path.join(clf_dir, f'test_predictions_{model}{suffix}.npz')
    np.savez_compressed(path, cells=np.asarray(cells, dtype=str), y_true=np.asarray(y_true, dtype=str),
                        P=np.asarray(P, dtype=np.float32), classes=CLASSES, label=label,
                        step=step, name=name, suffix=suffix)
    return path


def load_predictions(path):
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    assert list(d['classes']) == list(CLASSES), f"{path}: class order differs from CLASSES"
    for k in ('label', 'step', 'name', 'suffix'):
        d[k] = str(d[k])
    d['pred'] = CLASSES[d['P'].argmax(axis=1)]
    return d


def confusion_by_true(y_true, pred, title, path):
    """Confusion matrix as the fraction of each true class (rows sum to 1): the diagonal is recall."""
    cm = pd.crosstab(pd.Series(y_true, name='True'), pd.Series(pred, name='Predicted'), normalize='index') \
        .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                cbar_kws={'label': 'Fraction of true class'}, ax=ax)
    ax.set(title=f'{title}: test set\n{score_text(y_true, pred)}\nrows sum to 1; diagonal = recall')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_by_predicted(y_true, pred, title, path):
    """Confusion matrix as the fraction of each predicted class (columns sum to 1): the diagonal is
    precision, and the rest of each column shows what the wrong calls really were."""
    cm = pd.crosstab(pd.Series(y_true, name='True'), pd.Series(pred, name='Predicted'), normalize='columns') \
        .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                cbar_kws={'label': 'Fraction of cells predicted as that class'}, ax=ax)
    ax.set(title=f'{title}: test set\n{score_text(y_true, pred)}\ncolumns sum to 1; diagonal = precision')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def precision_recall_bars(y_true, pred, title, path):
    """Precision and recall of each class, side by side, with each class's number of test cells."""
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_true, pred, labels=CLASSES, average=None, zero_division=0)
    rec = recall_score(y_true, pred, labels=CLASSES, average=None, zero_division=0)
    n = pd.Series(y_true).value_counts().reindex(CLASSES).fillna(0).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x, w = np.arange(len(CLASSES)), 0.38
    bars_p = ax.bar(x - w / 2, prec, w, label='Precision', color='#2b6cb0')
    bars_r = ax.bar(x + w / 2, rec, w, label='Recall', color='#9ac0e6')
    ax.bar_label(bars_p, fmt='%.2f', fontsize=7)
    ax.bar_label(bars_r, fmt='%.2f', fontsize=7)
    ax.set_xticks(x, [f'{c}\n(n={k:,})' for c, k in zip(CLASSES, n)], fontsize=8)
    ax.set(ylim=(0, 1.08), ylabel='Test set', title=f'{title}: precision and recall\n{score_text(y_true, pred)}')
    ax.legend(loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_model_figures(pred_path, fig_dir):
    """Draw a model's three test figures from its predictions file into fig_dir; returns their paths."""
    d = load_predictions(pred_path)
    stem = os.path.join(fig_dir, f"{d['name']}test_")
    paths = [f"{stem}confusion_matrix{d['suffix']}.png", f"{stem}confusion_matrix_precision{d['suffix']}.png",
             f"{stem}precision_recall{d['suffix']}.png"]
    for draw, path in zip([confusion_by_true, confusion_by_predicted, precision_recall_bars], paths):
        draw(d['y_true'], d['pred'], d['label'], path)
    return paths
